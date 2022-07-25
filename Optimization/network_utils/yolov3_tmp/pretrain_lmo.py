import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import random as random

import argparse
from sklearn import svm, linear_model
from tensorboardX import SummaryWriter
import os, sys
import torch
from datetime import datetime
# from lts_utils.lts import lts, lts_yolo
# from influence_utils.influence import influence_gmm
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
from bayes_opt import BayesianOptimization
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

SEED = 100
np.random.seed(SEED)
random.seed(SEED)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

from train_lts import train_yolo_lts
from yolo_utils.datasets import *
from yolo_utils import torch_utils
import torch



# torch.backends.cudnn.enabled = False
qual_list=['high','low']
index_list=['30_1000','30_100']

# for qual in qual_list:
# 	for index in index_list:
qual='high'
index='30_1000'
cfg=1
device=str(0)
opt_weights=1
scene=str(2)

if cfg==0:
    net_cfg='cfg/yolov3-tiny-lmo8.cfg'
elif cfg==1:
    net_cfg='cfg/yolov3-spp-lmo8.cfg'
elif cfg==2:
    net_cfg='cfg/yolov3-lmo8.cfg'
elif cfg==3:
    net_cfg='cfg/csresnext50-panet-spp-lmo8.cfg'
if opt_weights==0:
    weights='weights/best_scene1_hightest_list_10_20.pt'
elif opt_weights==1:
    weights='weights/ultralytics68.pt'
elif opt_weights==2:
    weights='weights/yolov3-tiny.weights'
elif opt_weights==3:
    weights='weights/best_scene1_hightest_list_10_20.pt'
run_name=qual+'_'+index+'_'+str(cfg)+'_'+str(opt_weights)+'_'+scene
print(run_name)
print('qual is', qual)
print('index is', index)
print('cfg is', cfg)
print('weights is', weights)
print('scene is', scene)
train_yolo_lts_opt = Namespace(accumulate=4, adam=False, arc='default', batch_size=12, bucket='', cache_images=False, cfg=net_cfg, data='lmopre.data', device=device, epochs=273, evolve=False, img_size=[416], multi_scale=False, name=run_name, nosave=True, notest=False, rect=False, resume=False, single_cls=False, train_txt='../../../icml_experiments_data/scene%s_%s/test_list_%s.txt'%(scene,qual,index), var=None, weights=weights, freeze_backbone=False, cutoff=0, it=0, save_location='../../runs/pre_training')
print(train_yolo_lts_opt.train_txt,train_yolo_lts_opt.name,train_yolo_lts_opt.device)
train_yolo_lts(train_yolo_lts_opt)


