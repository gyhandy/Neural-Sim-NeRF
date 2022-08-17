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


from latentfusion.recon.inference import Observation
from latentfusion.pose import bop
from latentfusion.datasets.bop import BOPDataset
from latentfusion.datasets.realsense import RealsenseDataset
import latentfusion.visualization as viz
from latentfusion.augment import gan_denormalize
from latentfusion import meshutils
from latentfusion import augment

LINEMOD_ID_TO_NAME = {
    '000001': 'ape',
    '000002': 'benchvise',
    '000003': 'bowl',
    '000004': 'camera',
    '000005': 'can',
    '000006': 'cat',
    '000007': 'mug',
    '000008': 'driller',
    '000009': 'duck',
    '000010': 'eggbox',
    '000011': 'glue',
    '000012': 'holepuncher',
    '000013': 'iron',
    '000014': 'lamp',
    '000015': 'phone',
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--object_id', type=int)
args = parser.parse_args()

object_id = args.object_id
print(object_id)
skip_every = 1
object_name = LINEMOD_ID_TO_NAME[f"{object_id:06d}"]
dataset_dir = NERF_PATH / object_name
if not os.path.exists(dataset_dir):
    dataset_dir.mkdir()

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def write_transform(filepath, dataset):
    near, far = dataset.camera[0].z_bounds
    num_images = len(dataset.camera)

    frames = []
    for i in range(num_images):
        frame = {}
        frame['file_path'] = str(dataset.color_paths[i])
        pose_openCV = np.linalg.inv(dataset.camera.extrinsic[i].numpy())
        pose_openGL = convert_pose(pose_openCV).tolist()
        frame['transform_matrix'] = pose_openGL
        intrinsic = dataset.camera.intrinsic[i].numpy().tolist()
        frame['intrinsic_matrix'] = intrinsic
        frames.append(frame)

    data = {}
    data['near'] = near.numpy()[0] - 0.25
    data['far'] = far.numpy()[0] + 0.25
    data['frames'] = frames
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def build_dataset(split):
    print(f"Building {split} dataset ...")
    scene_path = LINEMOD_PATH / f'{split}/{object_id:06d}'
    dataset = BOPDataset(LINEMOD_PATH, scene_path, object_id=object_id, object_scale=None)
    obs = Observation.from_dataset(dataset)
    obs.color_paths = dataset.color_paths

    write_transform(str(dataset_dir / f'transforms_{split}.json'), obs)

for split in ['train', 'test']:
    build_dataset(split)

# Create dummy val json files by copying train
from shutil import copyfile
copyfile(dataset_dir / 'transforms_train.json', dataset_dir / 'transforms_val.json')