'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import copy
from datetime import datetime
import os
from pathlib import Path
import gzip
import pickle
import numpy as np
import open3d as o3d
import torch
from tqdm import trange

import hydra
import omegaconf
from omegaconf import DictConfig

# Local application/library specific imports
from openfungraph.dataset.datasets_common import get_dataset
from openfungraph.utils.vis import OnlineObjectRenderer
from openfungraph.utils.ious import (
    compute_2d_box_contained_batch
)
from openfungraph.slam.slam_classes import MapObjectList
from openfungraph.slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
)
from openfungraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

BG_CLASSES = ["wall", "floor", "ceiling"]

# Disable torch gradient computation
torch.set_grad_enabled(False)

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.01):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()
    
def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg
    
@hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def main(cfg : DictConfig):
    cfg = process_cfg(cfg)
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()
    
    classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)

    objects = MapObjectList(device=cfg.device)

    if not cfg.skip_bg:
        # Handle the background detection separately
        # Each class of them are fused into the map as a single object
        bg_objects = {
            c: None for c in BG_CLASSES
        }
    else:
        bg_objects = None

    viser_server = None
    frame_handles = []
    object_handles = {}
    point_size_slider = None
    if cfg.get("use_viser", False):
        try:
            import viser

            viser_server = viser.ViserServer()
            point_size_slider = viser_server.gui.add_slider(
                label="point_size", min=0.001, max=0.05, step=0.001, initial_value=0.01
            )
        except Exception as e:
            print(f"Failed to launch viser server: {e}")
            cfg.use_viser = False

        def update_viser(frame_idx, fg_list, bg_list, obj_list):
            nonlocal frame_handles, object_handles, point_size_slider

            # Hide previous frame detections
            for h in frame_handles:
                h.visible = False
            frame_handles = []

            point_size = point_size_slider.value if point_size_slider else 0.01
            frame_ns = f"frame/frame_{frame_idx}"
            for det_list, name in ((fg_list, "fg_detection"), (bg_list, "bg_detection")):
                for det_idx, det in enumerate(det_list):
                    pts = np.asarray(det["pcd"].points)
                    if pts.size == 0:
                        continue
                    cid = det["class_id"][0] if isinstance(det["class_id"], list) else det["class_id"]
                    col = np.asarray(class_colors[str(cid)])
                    cols = np.tile(col, (pts.shape[0], 1))
                    handle = viser_server.add_point_cloud(
                        f"{frame_ns}/{name}/{det_idx}",
                        points=pts,
                        colors=cols,
                        point_size=point_size,
                    )
                    frame_handles.append(handle)

            # Remove handles of objects that no longer exist
            for path in list(object_handles.keys()):
                idx = int(path.split("_")[-1])
                if idx >= len(obj_list):
                    object_handles[path].visible = False
                    del object_handles[path]

            # Update or create handles for current objects
            for i, obj in enumerate(obj_list):
                pts = np.asarray(obj["pcd"].points)
                if pts.size == 0:
                    continue
                cid = obj["class_id"][0] if isinstance(obj["class_id"], list) else obj["class_id"]
                col = np.asarray(class_colors[str(cid)])
                cols = np.tile(col, (pts.shape[0], 1))
                path = f"object/object_{i}"
                if path in object_handles:
                    handle = object_handles[path]
                    handle.points = pts
                    handle.colors = cols
                    handle.point_size = point_size
                    handle.visible = True
                else:
                    handle = viser_server.add_point_cloud(
                        path, points=pts, colors=cols, point_size=point_size
                    )
                    object_handles[path] = handle

    for idx in trange(len(dataset)):
        # get color image
        color_path = dataset.color_paths[idx]

        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        # image_rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        # Get the RGB image
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"
        
        # Get the depth image
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()

        # Get the intrinsics matrix
        cam_K = intrinsics.cpu().numpy()[:3, :3]
        
        # load grounded SAM detections
        gobs = None # stands for grounded SAM observations

        color_path = Path(color_path)
        if not cfg.part_reg:
            detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
        else:
            detections_path = color_path.parent.parent / 'part' /cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
        color_path = str(color_path)
        detections_path = str(detections_path)
        
        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)


        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[idx]
        unt_pose = unt_pose.cpu().numpy()
        
        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = cfg,
            image = image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = adjusted_pose,
            class_names = classes,
            BG_CLASSES = BG_CLASSES,
            color_path = color_path,
            part_reg = cfg.part_reg,
        )

        if cfg.get("use_viser", False) and len(fg_detection_list) == 0:
            update_viser(idx, fg_detection_list, bg_detection_list, objects)

        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            
        if len(fg_detection_list) == 0:
            continue
            
        if cfg.use_contain_number:
            xyxy = fg_detection_list.get_stacked_values_torch('xyxy', 0)
            contain_numbers = compute_2d_box_contained_batch(xyxy, cfg.contain_area_thresh)
            for i in range(len(fg_detection_list)):
                fg_detection_list[i]['contain_number'] = [contain_numbers[i]]
            
        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])

            # Skip the similarity computation 
            continue
                
        spatial_sim = compute_spatial_similarities(cfg, fg_detection_list, objects)
        visual_sim = compute_visual_similarities(cfg, fg_detection_list, objects)
        agg_sim = aggregate_similarities(cfg, spatial_sim, visual_sim)
        
        # Compute the contain numbers for each detection
        if cfg.use_contain_number:
            # Get the contain numbers for all objects
            contain_numbers_objects = torch.Tensor([obj['contain_number'][0] for obj in objects])
            detection_contained = contain_numbers > 0 # (M,)
            object_contained = contain_numbers_objects > 0 # (N,)
            detection_contained = detection_contained.unsqueeze(1) # (M, 1)
            object_contained = object_contained.unsqueeze(0) # (1, N)                

            # Get the non-matching entries, penalize their similarities
            xor = detection_contained ^ object_contained
            agg_sim[xor] = agg_sim[xor] - cfg.contain_mismatch_penalty
        
        # Threshold sims according to cfg. Set to negative infinity if below threshold
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')

        objects = merge_detections_to_objects(cfg, fg_detection_list, objects, agg_sim)

        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (idx+1) % cfg.denoise_interval == 0:
            objects = denoise_objects(cfg, objects)
        if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
            objects = filter_objects(cfg, objects)
        if cfg.merge_interval > 0 and (idx+1) % cfg.merge_interval == 0:
            objects = merge_objects(cfg, objects)

        if cfg.get("use_viser", False):
            update_viser(idx, fg_detection_list, bg_detection_list, objects)

    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        bg_objects = denoise_objects(cfg, bg_objects)
        
    objects = denoise_objects(cfg, objects)
    
    # Save the full point cloud before post-processing
    if cfg.save_pcd:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
            'class_names': classes,
            'class_colors': class_colors,
        }

        if not cfg.part_reg:
            pcd_save_path = cfg.dataset_root / \
                cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        else:
            pcd_save_path = cfg.dataset_root / \
                cfg.scene_id / 'part' / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        # make the directory if it doesn't exist
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        pcd_save_path = str(pcd_save_path)
        
        # with gzip.open(pcd_save_path, "wb") as f:
        #     pickle.dump(results, f)
        # print(f"Saved full point cloud to {pcd_save_path}")
    
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    
    # Save again the full point cloud after the post-processing
    if cfg.save_pcd:
        results['objects'] = objects.to_serializable()
        pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud after post-processing to {pcd_save_path}")
        
if __name__ == "__main__":
    main()