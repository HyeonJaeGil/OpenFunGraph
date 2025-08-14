# Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces

[**Project Page**](https://openfungraph.github.io/)

![Splash Figure Top](./assets/teaser_top.jpg)
![Splash Figure](./assets/teaser.png)

## Setup

### Install the required libraries

```bash
conda create -n openfungraph python=3.10
conda activate openfungraph

##### Install Pytorch according to your own setup #####
# For example, if you have a GPU with CUDA 11.8
# Note that this version is compatible with the LLaVA repo
# Here we install cudatoolkit via Conda for installation of Grounded-SAM
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia

# Install the Faiss library (CPU version should be fine)
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

# Install Pytorch3D by 
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py38_cu118_pyt212.tar.bz2

# Install the required libraries
pip install tyro open_clip_torch==2.18.0 wandb h5py openai hydra-core distinctipy pyviz3d line_profiler
pip install "opencv-python<4.10" "numpy==1.26.4" "open3d==0.18.0"

# Install the gradslam package and its dependencies
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
pip install .
cd ..
git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
pip install .
```

### Install [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) package

Follow the instructions on the original [repo](https://github.com/IDEA-Research/Grounded-Segment-Anything#install-without-docker). 

First checkout the package by 

```bash
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
```

Then, install the package Following the commands listed in the original GitHub repo. You can skip the `Install osx` step and the "optional dependencies". 

During this process, you will need to set the `CUDA_HOME` to be where the CUDA toolkit is installed. 
The CUDA tookit can be set up system-wide or within a conda environment. 
We tested it within a conda environment, i.e. installing [cudatoolkit-dev](https://anaconda.org/conda-forge/cudatoolkit-dev) using conda by former commands. 

```bash
# and you need to replace `export CUDA_HOME=/path/to/cuda-11.3/` by 
export CUDA_HOME=/path/to/anaconda3/envs/openfungraph/
```

You also need to download `ram_swin_large_14m.pth`, `groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth` following the instruction [here](https://github.com/IDEA-Research/Grounded-Segment-Anything#label-grounded-sam-with-ram-or-tag2text-for-automatic-labeling). 

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.8
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

After installation, set the path to Grounded-SAM as an environment variable.

```bash
export GSA_PATH=/path/to/Grounded-Segment-Anything
```

### Set up LLaVA

Follow the instructions on the [LLaVA repo](https://github.com/haotian-liu/LLaVA) to set it up. We have tested with model checkpoint `LLaVA-7B-v1.6`.

```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA/
pip install -e .
```

### Install this repo

```bash
cd OpenFunGraph
pip install -e .
```

## Prepare dataset

Download the [customized SceneFun3D dataset](https://huggingface.co/datasets/OpenFunGraph/SceneFun3D_Graph) and the newly recorded [FunGraph3D dataset](https://huggingface.co/datasets/OpenFunGraph/FunGraph3D). 

In their top repo, file structure is introduced.
Note that related path should be set as ``env_vars.bash.template``.

```bash
export FUNGRAPH3D_ROOT=
export FUNGRAPH3D_CONFIG_PATH=${FG_FOLDER}/openfungraph/dataset/dataconfigs/fungraph3d/fungraph3d.yaml
export SCENEFUN3D_ROOT=  # for SceneFun3D, it should be with dev / test
export SCENEFUN3D_CONFIG_PATH=${FG_FOLDER}/openfungraph/dataset/dataconfigs/scenefun3d/scenefun3d.yaml
```

OpenFunGraph can also be easily run on other dataset. 
See `dataset/datasets_common.py` for how to write your own dataloader. 

## Run OpenFunGraph

The env variables needed can be found in `env_vars.bash.template`. 
When following the setup guide below, you should change the variables accordingly for easy setup. 

The following commands should be run in the `openfungraph` folder.

```bash
cd openfungraph
```

### Functional Scene Graph Node Detection

```bash
export SCENE_NAME=

bash scenegraph/detection_scenefun3d.sh (or *_fungraph3d.sh)
```

The above commands will save the 2D node detection and segmentation results.

You can ignore the `There's a wrong phrase happen, this is because of our post-process merged wrong tokens, which will be modified in the future. We will assign it with a random label at this time.` message. 

### 3D Functional Scene Graph Construction

Ensure that the `openai` package is installed and that your APIKEY is set. We recommend using GPT-4.
```bash
export OPENAI_API_KEY=<your GPT-4 API KEY here>
```

```bash
CUDA_VISIBLE_DEVICES=0 python scenegraph/build_fungraph_whole_openai.py --dataset_root ${SCENEFUN3D_ROOT} ``or`` ${FUNGRAPH3D_ROOT}  --scene_name ${SCENE_NAME} --mapfile ``<SCENE_PATH>/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz`` --part_file ``<SCENE_PATH>/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz`` 
```

### Visualize and Evaluation

After running the algorithm, you can get three modified key assets of object-level nodes ``<SCENE_PATH>/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.3_bbox0.9_simsum1.2_dbscan.1_post.pkl.gz`` (the name could be varied depending on what parameters you choose), sub-object-level elements ``<SCENE_PATH>/part/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.15_bbox0.1_simsum1.2_dbscan.1_parts_post.pkl.gz``, and the finel graph edges ``<SCENE_PATH>/cfslam_funcgraph_edges.pkl`` (or with confidence).

Visualize them by
```bash
python scripts/pyviz3d_interactable_results.py --inter_result_path <PATH TO OBJECT-LEVEL NODES> --part_result_path <PATH TO SUB-OBJECT-LEVEL ELEMENTS> --edge_file <PATH TO GRAPH EDGES> --pc_path <PATH TO SCENE POINT CLOUD> (--pose_path (only for SCENEFUN3D) <SCENE_PATH>/*_transform.npy) 
```

Evaluation scripts:
For node evaluation
```bash
python eval/eval_node.py --dataset <SceneFun3D or FunGraph3D> --root_path <PATH TO THE DATASET> --scene <SCENE NAME> --video <VIDEO NAME> (--split (only for SCENEFUN3D) <dev or test>)
```

You can also control the top K value for retrieval and the IoU threshold for spatial alignment by using ``--topk`` and ``--iou_threshold``. 

For triplet evaluation
```bash
python eval/eval_triplet.py --dataset <SceneFun3D or FunGraph3D> --root_path <PATH TO THE DATASET> --scene <SCENE NAME> --video <VIDEO NAME> (--split (only for SCENEFUN3D) <dev or test>)
```