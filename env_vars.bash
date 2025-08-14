#!/bin/bash
# Example script setting up the rnv variables needed for running OpenFunGraph
# Please adapt it to your own paths!

# cd openfungraph

# conda activate openfungraph

export FG_FOLDER=${PWD}
export GSA_PATH=${PWD}/Grounded-Segment-Anything
export FUNGRAPH3D_ROOT=/mnt/Backup2nd/SceneFun3D_Graph/FunGraph3D/
export FUNGRAPH3D_CONFIG_PATH=${FG_FOLDER}/openfungraph/dataset/dataconfigs/fungraph3d/fungraph3d.yaml
export SCENEFUN3D_ROOT=/mnt/Backup2nd/SceneFun3D_Graph/SceneFun3D_Graph/dev/  # for SceneFun3D, it should be with dev / test
export SCENEFUN3D_CONFIG_PATH=${FG_FOLDER}/openfungraph/dataset/dataconfigs/scenefun3d/scenefun3d.yaml
export SCENE_NAME=420683/42445132/
export THRESHOLD=1.2
export CLASS_SET=ram

export OPENAI_API_KEY=
