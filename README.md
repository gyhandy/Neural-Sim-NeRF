# Neural-Sim: Learning to Generate Training Data with NeRF

[ECCV 2022] Neural-Sim: Learning to Generate Training Data with NeRF

The code is actively updating, thanks!

## Overview
The code is trying to use generative models to boost the performance of discriminative models.
Specifically, we use data to train a nerf and we want to obtain the **optimal parameter** that can
use nerf to synthesize more demanded data, which can be used to train a down stream detection model. 
We use bi-level optimization to optimize the Nerf parameter with the loss on validation set.

It contains 4 mian parts:

1 BlenderProc to synthesize controllable object images (e.g., YCB, LMO)

2 Nerf training

3 Detectron2 training and testing

4 End-to-end Bilevel Optimization


### 0 install

git pull Neural-Sim

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
