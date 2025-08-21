# generate object 2D detection
CUDA_VISIBLE_DEVICES=0 python scripts/generate_gsa_results.py mode=scene dataset_root=$FUNGRAPH3D_ROOT dataset_config=$FUNGRAPH3D_CONFIG_PATH scene_id=$SCENE_NAME class_set=$CLASS_SET box_threshold=0.25 text_threshold=0.25 stride=1 add_bg_classes=True accumu_classes=True exp_suffix=withbg_allclasses
# fuse general objects
python slam/cfslam_pipeline_batch.py     dataset_root=$FUNGRAPH3D_ROOT     dataset_config=$FUNGRAPH3D_CONFIG_PATH     stride=1     scene_id=$SCENE_NAME     spatial_sim_type=overlap     mask_conf_threshold=0.3     match_method=sim_sum     sim_threshold=${THRESHOLD}     dbscan_eps=0.1     gsa_variant=ram_withbg_allclasses     skip_bg=False     max_bbox_area_ratio=0.9    merge_overlap_thresh=0.9 save_suffix=overlap_maskconf0.3_bbox0.9_simsum${THRESHOLD}_dbscan.1 merge_visual_sim_thresh=0.75 merge_text_sim_thresh=0.7

# detect 2D parts
CUDA_VISIBLE_DEVICES=0 python scripts/generate_gsa_results.py mode=part dataset_root=$FUNGRAPH3D_ROOT dataset_config=$FUNGRAPH3D_CONFIG_PATH scene_id=$SCENE_NAME class_set=$CLASS_SET box_threshold=0.15 text_threshold=0.15 stride=1 add_bg_classes=True accumu_classes=True exp_suffix=withbg_allclasses

# fuse parts
python slam/cfslam_pipeline_batch.py     dataset_root=$FUNGRAPH3D_ROOT     dataset_config=$FUNGRAPH3D_CONFIG_PATH     stride=1     scene_id=$SCENE_NAME     spatial_sim_type=overlap     mask_conf_threshold=0.15     match_method=sim_sum     sim_threshold=${THRESHOLD}     dbscan_eps=0.1     gsa_variant=ram_withbg_allclasses     skip_bg=False     max_bbox_area_ratio=0.1     save_suffix=overlap_maskconf0.15_bbox0.1_simsum${THRESHOLD}_dbscan.1_parts part_reg=True

python scripts/ana_rigid_objs.py --result_path $FUNGRAPH3D_ROOT'/'$SCENE_NAME'/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz'  --part_result_path $FUNGRAPH3D_ROOT'/'$SCENE_NAME'/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz'

CUDA_VISIBLE_DEVICES=0 python scenegraph/build_fungraph_whole_openai.py --dataset_root ${FUNGRAPH3D_ROOT}  --scene_name ${SCENE_NAME} --mapfile ${FUNGRAPH3D_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz --part_file ${FUNGRAPH3D_ROOT}/${SCENE_NAME}/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz