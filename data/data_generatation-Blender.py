import os
from pathlib import Path
import numpy as np
import torch
import json
import imageio
from torch.backends import cudnn
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LINEMOD_PATH = Path('/BLENDERPROC/OUTPUT/PATH/bop_data/ycbv') # BlenderProc synthesized images
NERF_PATH = Path('/OUTPUT/DATA/PATH/TO/TRAIN/NERF')

Num_train = 200 # number of trianing images, the rest would be test image

from shutil import copyfile

LINEMOD_ID_TO_NAME = {
    '000001': 'coffee',
    '000002': 'cheesebox',
    '000003': 'sugerbox',
    '000010': 'banana',
    '000013': 'bowl',
    '000015': 'drill',
}

OBJECT_DIAMETER = {
    '000001': 0.18,
    '000002': 0.28,
    '000003': 0.2,
    '000010': 0.2,
    '000013': 0.17,
    '000015': 0.23,
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--object_id', default= 2, type=int)
args = parser.parse_args()

object_id = args.object_id
print(object_id)
skip_every = 1
object_name = LINEMOD_ID_TO_NAME[f"{object_id:06d}"]
# dataset_dir = NERF_PATH / object_name
dataset_dir = NERF_PATH
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
# create train test val image in target
splits = ['train','test']
for s in splits:
    folder = os.path.join(dataset_dir, s)
    if not os.path.exists(folder):
        os.makedirs(folder)

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

'''source'''
rgb_dir = os.path.join(LINEMOD_PATH, 'train_pbr', '000000', 'rgb')
imgs= sorted(os.listdir(rgb_dir))
# camera info
with open(os.path.join(LINEMOD_PATH, 'camera.json'), 'r') as fp:
    camera_info = json.load(fp)
# pose info
with open(os.path.join(LINEMOD_PATH, 'train_pbr', '000000', 'scene_camera.json'), 'r') as fp:
    pose_info = json.load(fp)

'''scale parameter'''
object_diameter = OBJECT_DIAMETER[f"{object_id:06d}"] # this is important to calculate near and far

'''have train and test'''
train_frames = []
train_near = []
train_far = []
test_frames = []
test_near = []
test_far = []
for i, img_path in tqdm(enumerate(imgs)):
    frame = {}
    # frame['file_path'] = str(dataset.color_paths[i])
    set_name = 'train' if i < Num_train else 'test'
    # frame['file_path'] = os.path.join(NERF_PATH, object_name, set_name, img_path) # use nerf dataset name
    frame['file_path'] = os.path.join(NERF_PATH, set_name, img_path)  # use nerf dataset name
    # save image
    copyfile(os.path.join(rgb_dir, img_path), frame['file_path']) # origin to target
    pose = pose_info[str(int(img_path.split('.')[0]))]
    pose_w2c_R = np.array(pose['cam_R_w2c'])
    pose_w2c_R.resize((3,3))
    pose_w2c_t = np.array(pose['cam_t_w2c'])
    pose_w2c_t.resize((3,1))

    # pose_w2c_t *= (basic_object_scale / object_diameter) # 0.006
    pose_w2c_t *= 0.001  # mm to m
    pose_w2c = np.concatenate((np.concatenate((pose_w2c_R, pose_w2c_t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
    pose_openCV = np.linalg.inv(pose_w2c)
    pose_openGL = convert_pose(pose_openCV).tolist()
    frame['transform_matrix'] = pose_openGL
    pose_K = np.array(pose['cam_K'])
    pose_K.resize((3,3))
    intrinsic = pose_K.tolist()
    frame['intrinsic_matrix'] = intrinsic
    if i < Num_train: # train data
        train_frames.append(frame)
        train_near.append(pose_w2c_t[-1]-object_diameter/2)
        train_far.append(pose_w2c_t[-1] + object_diameter / 2)
        # train_near.append(pose_w2c_t[-1]-object_diameter)
        # train_far.append(pose_w2c_t[-1] + object_diameter)
    else: # test data
        test_frames.append(frame)
        test_near.append(pose_w2c_t[-1]-object_diameter/2)
        test_far.append(pose_w2c_t[-1] + object_diameter / 2)
        # test_near.append(pose_w2c_t[-1]-object_diameter)
        # test_far.append(pose_w2c_t[-1] + object_diameter)

'''save train and test'''
train_data = {}
train_data['near'] = float(min(train_near) - 0.05) # 0.05 represent the enlarged margin
train_data['far'] = float(max(train_far)+ 0.05)
train_data['frames'] = train_frames
filepath = os.path.join(dataset_dir, 'transforms_train.json')
with open(filepath, 'w') as f:
    json.dump(train_data, f, indent=4)

test_data = {}
test_data['near'] = float(min(test_near) - 0.05)
test_data['far'] = float(max(test_far)+ 0.05)
test_data['frames'] = test_frames
filepath = os.path.join(dataset_dir, 'transforms_test.json')
with open(filepath, 'w') as f:
    json.dump(test_data, f, indent=4)

copyfile(dataset_dir / 'transforms_train.json', dataset_dir / 'transforms_val.json')
# copyfile(dataset_dir / 'transforms_train.json', dataset_dir / 'transforms_test.json')
