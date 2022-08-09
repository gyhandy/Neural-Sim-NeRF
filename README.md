# Neural-Sim: Learning to Generate Training Data with NeRF

[ECCV 2022] [Neural-Sim: Learning to Generate Training Data with NeRF](https://arxiv.org/pdf/2207.11368.pdf)

Code are actively updating, thanks!

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
    <img src="./docs/pipeline.png" alt="Editor" width="800">
</div>

## 1 Installation

### Getting started

Start by cloning the repo:

<<<<<<< HEAD
```bash
git clone https://github.com/gyhandy/Neural-Sim-NeRF.git
```


1 install the requirement of nerf-pytorch
```bash
pip install -r requirements.txt
```

2 install [detectorn2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
=======

### 1 Nerf training 

Train nerf with instructions (https://github.com/yenchenlin/nerf-pytorch)



### 2 Detectron2 and train the models

Create a coco format dataset and then train detectron2 with the prepared dataset.
>>>>>>> 0ac20eabd9625976ad74ba495abdbaa8ef1e5620


### 3 Bilelve optimization pipeline

```bash
cd ./optimization
```

Please use the Nerf_AutoSimulate_differentiable_general_all.py to run the end-to-end pipeline.

```bash
python neural_sim_main.py --config ../configs/cheese_noscale_auto_general.txt --object_id 15 --expname ycbv6c-15-bin3 --psi_pose_cats_mode 3
```
