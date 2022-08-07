# Neural-Sim: Learning to Generate Training Data with NeRF

[ECCV 2022] [Neural-Sim: Learning to Generate Training Data with NeRF](https://arxiv.org/pdf/2207.11368.pdf)


## Overview
The code is for On-demand synthetic data generation: Given a target task and a
test dataset, our approach “Neural-sim” generates data on-demand using a fully
differentiable synthetic data generation pipeline which maximises accuracy for
the target task.
<div align="center">
    <img src="./docs/neural-sim.png" alt="Editor" width="500">
</div>


Neural-Sim pipeline: Our pipeline finds the optimal parameters for generating views from a trained neural renderer (NeRF) to use as training data for
object detection. The objective is to find the optimal NeRF rendering parameters ψ that can generate synthetic training data Dtrain, such that the model
(RetinaNet, in our experiments) trained on Dtrain, maximizes accuracy on a
downstream task represented by the validation set Dval

<div align="center">
    <img src="./docs/pipeline.png" alt="Editor" width="500">
</div>

### 0 install

git clone Neural-Sim-NeRF

1 install the requirement of nerf-pytorch
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt

2 install detectorn 2
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
find the correct version based on cuda and pytorch


### 1 Generate Bop format images with BlenderProc 
```shell
cd ./BlenderProc
```
Install necessary environment as (https://github.com/DLR-RM/BlenderProc)

Follow the examples (https://github.com/DLR-RM/BlenderProc/blob/main/README.md#examples)
to understand the basic configuration file meaning.

We use custom configure files in ./examples/camera_sampling

Note: 
1 Create a new environment for Blenderproc synthesis to install Bop_toolkit
2 the configuration of BOP dataset should be carefor:
unzip ycbv_base.zip get the ycbv folder, unzip ycbv_models.zip get the models folder, move the models folder into ycbv folder

example command
```shell
python run.py examples/camera_sampling/config.yaml ~/PycharmProject/data/BOP/ ycbv ~/PycharmProject/data/BOP/bop_toolkit/ examples/YCBV/ycbvid3-1000-r15
```

or
python run.py examples/camera_sampling/config_same.yaml /data/BOP ycbv /data/BOP/bop_toolkit/ examples/YCBV/ycbvid3-1000-r15
python run.py examples/camera_sampling/config2.yaml /data/BOP ycbv /data/BOP/bop_toolkit/ /data/BOP/YCBV_nerf_train_data/ycbvid11-300-r1-P123
2 Process synthesized images to be admitted by nerf (OPENCV --> OPENGL)

if use no scale, this is the current data-generation code for cvpr and eccv.
```shell
python data_generation-Blender.py
``` 


if use LatentFusion read BOP format data
```shell 
python data_generation-LINEMOD.py
 ```
### 2 Nerf training 

Train nerf with instructions (https://github.com/yenchenlin/nerf-pytorch)



### 3 Detectron2 and train the models

Create a coco format dataset and then train detectron2 with the prepared dataset.


### 4 Bilelve optimization pipeline

```shell
cd ./Optimization
```

Please use the Nerf_AutoSimulate_differentiable_general_all.py to run the end-to-end pipeline.

```shell
python Nerf_AutoSimulate_differentiable_general_all.py --config ../configs/cheese_noscale_auto_general.txt --object_id 15 --expname ycbv6c-15-bin3 --psi_pose_cats_mode 3
```
