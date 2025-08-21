'''
The script is used to extract Grounded SAM results on a posed RGB-D dataset.
It can generate either the full scene results or the part-based results.
The results will be dumped to a folder under the scene folder.
Use ``--mode part`` to enable part-based processing.
'''

import os
from pathlib import Path
from typing import Any, List
from PIL import Image
import cv2
import json
import imageio
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pickle
import gzip
import open_clip
import torch
import torchvision
import supervision as sv
from tqdm import trange

import hydra
import omegaconf
from omegaconf import DictConfig

from openfungraph.dataset.datasets_common import get_dataset
from openfungraph.utils.vis import vis_result_fast, vis_result_slow_caption
from openfungraph.utils.model_utils import compute_clip_features

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
sys.path.append(GSA_PATH) # This is needed for the following imports in this file

import torchvision.transforms as TS
try:
    from ram.models import ram
    from ram import inference_ram
except ImportError as e:
    print("RAM sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg


# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_sam_predictor(device: str | int) -> SamPredictor:

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor
    

# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
   
    results = model.generate(image)
    mask = []
    xyxy = []
    conf = []
    for r in results:
        mask.append(r["segmentation"])
        r_xyxy = r["bbox"].copy()
        # Convert from xyhw format to xyxy format
        r_xyxy[2] += r_xyxy[0]
        r_xyxy[3] += r_xyxy[1]
        xyxy.append(r_xyxy)
        conf.append(r["predicted_iou"])
    mask = np.array(mask)
    xyxy = np.array(xyxy)
    conf = np.array(conf)
    return mask, xyxy, conf


def get_sam_mask_generator(device: str | int) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=12,
        points_per_batch=144,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    return mask_generator


def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes

    
@hydra.main(version_base=None, config_path="../configs/slam_pipeline", config_name="base")
def main(cfg: DictConfig):
    cfg = process_cfg(cfg)
    ### Initialize the Grounding DINO model ###
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=cfg.device
    )

    ### Initialize the SAM model ###
    if cfg.class_set == "none":
        mask_generator = get_sam_mask_generator(cfg.device)
    else:
        sam_predictor = get_sam_predictor(cfg.device)

    ###
    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(cfg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

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

    global_classes = set()

    if cfg.mode != "part" and cfg.class_set in ["ram"]:

        tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                        image_size=384,
                                        vit='swin_l')

        tagging_model = tagging_model.eval().to(cfg.device)

        # initialize Tag2Text
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

        classes = None
    elif cfg.class_set == "none":
        classes = ['item']
    elif cfg.mode == "part":
        classes = None
    else:
        raise ValueError("Unknown class_set: ", cfg.class_set)

    if cfg.class_set == "none":
        print("Skipping tagging and detection models. ")
    elif cfg.class_set not in ["ram"]:
        print("There are total", len(classes), "classes to detect. ")
    else:
        print(f"{cfg.class_set} will be used to detect classes. ")

    save_name = f"{cfg.class_set}"
    if cfg.exp_suffix:
        save_name += f"_{cfg.exp_suffix}"

    for idx in trange(len(dataset)):
        ### Relevant paths and load image ###
        color_path = dataset.color_paths[idx]

        color_path = Path(color_path)

        if cfg.mode == "part":
            vis_save_path = color_path.parent.parent / "part" / f"gsa_vis_{save_name}" / color_path.name
            detections_save_path = color_path.parent.parent / "part" / f"gsa_detections_{save_name}" / color_path.name
        else:
            vis_save_path = color_path.parent.parent / f"gsa_vis_{save_name}" / color_path.name
            detections_save_path = color_path.parent.parent / f"gsa_detections_{save_name}" / color_path.name
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")

        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)

        # opencv can't read Path objects... sigh...
        color_path = str(color_path)
        vis_save_path = str(vis_save_path)
        detections_save_path = str(detections_save_path)

        image = cv2.imread(color_path) # This will in BGR color space
        # rotate it
        if hasattr(dataset, 'camera_axis'):
            if dataset.camera_axis == 'Left':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
        image_pil = Image.fromarray(image_rgb)
        caption = None
        text_prompt = None

        if cfg.mode == "part":
            add_classes = ["handle", "button", "knob", "drawer", "door", "oven", "closet", "window", "remote", "radiator", "bathhub", "sink"]
            classes = add_classes
        elif cfg.class_set in ["ram"]:
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(cfg.device)

            if cfg.class_set == "ram":
                res = inference_ram(raw_image , tagging_model)
                caption="NA"

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            text_prompt=res[0].replace(' |', ',')

            add_classes = ["other item", "door", "window", "drawer", "closet", "chest", "cabinet", "dresser", "radiator", "remote", "electric outlet", "trashcan"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium",
                "apartment", "image", "city", "blue", "skylight", "hallway",
                "bureau", "modern", "salon", "doorway", "wooden floor", "bedroom", "tile", "wood wall",
                "wall paper", "wallpaper", "polka dot", "dormitory", "hardwood floor", "wood floor",
                "mattress", "carpet", "plain", "sheet", "subway", "sun", "glass wall", "glass floor", "wall lamp", "glass door",
                "screen door", "hard wood", "ceiling", "grass", "close-up", "basement", "cement", "molding", "socket",
                "wood", "hardwood", "carpet", "rug", "heater", "concrete", "wall clock", "corridor"
            ]
            bg_classes = ["wall", "floor"]

            if cfg.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            classes = process_tag_classes(
                text_prompt,
                add_classes = add_classes,
                remove_classes = remove_classes,
            )

        # add classes to global classes
        global_classes.update(classes)

        if cfg.accumu_classes:
            # Use all the classes that have been seen so far
            classes = list(global_classes)

        ### Detection and segmentation ###
        if cfg.class_set == "none":
            # Directly use SAM in dense sampling mode to get segmentation
            mask, xyxy, conf = get_sam_segmentation_dense(mask_generator, image_rgb)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, cfg.device)

            ### Visualize results ###
            annotated_image, labels = vis_result_fast(
                image, detections, classes, instance_random_color=True)

            cv2.imwrite(vis_save_path, annotated_image)
        else:
            if cfg.detector == "dino":
                # Using GroundingDINO to detect and SAM to segment
                detections = grounding_dino_model.predict_with_classes(
                    image=image, # This function expects a BGR image...
                    classes=classes,
                    box_threshold=cfg.box_threshold,
                    text_threshold=cfg.text_threshold,
                )

                if len(detections.class_id) > 0:
                    ### Non-maximum suppression ###
                    # print(f"Before NMS: {len(detections.xyxy)} boxes")
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy),
                        torch.from_numpy(detections.confidence),
                        cfg.nms_threshold
                    ).numpy().tolist()
                    # print(f"After NMS: {len(detections.xyxy)} boxes")

                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]

                    # Somehow some detections will have class_id=-1, remove them
                    valid_idx = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid_idx]
                    detections.confidence = detections.confidence[valid_idx]
                    detections.class_id = detections.class_id[valid_idx]

            if len(detections.class_id) > 0:

                ### Segment Anything ###
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                # Compute and save the clip features of detections
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, cfg.device)
            else:
                image_crops, image_feats, text_feats = [], [], []

            ### Visualize results ###
            annotated_image, labels = vis_result_fast(image, detections, classes)

            # save the annotated grounded-sam image
            cv2.imwrite(vis_save_path, annotated_image)

        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }

        if cfg.mode != "part" and cfg.class_set in ["ram"]:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt

        # save the detections using pickle
        # Here we use gzip to compress the file, which could reduce the file size by 500x
        with gzip.open(detections_save_path, "wb") as f:
            pickle.dump(results, f)

    # save global classes
    if cfg.mode == "part":
        class_dir = cfg.dataset_root / cfg.scene_id / "part"
    else:
        class_dir = cfg.dataset_root / cfg.scene_id
    os.makedirs(class_dir, exist_ok=True)
    with open(class_dir / f"gsa_classes_{save_name}.json", "w") as f:
        json.dump(list(global_classes), f)


if __name__ == "__main__":
    main()

