import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import random as random

from sklearn import svm
import subprocess
import torch.optim as optim

import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import glob

from network_utils.datasets import Dataset_1dobject
import shutil
import os
import time
import subprocess 
import threading
import sys

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def on_timeout(proc, status_dict):
	"""Kill process on timeout and note as status_dict['timeout']=True"""
	# a container used to pass status back to calling thread
	status_dict['timeout'] = True
	# print("timed out")
	proc.kill()

def gen_gmm_dataset(params, dataset_size=100):
	comp=[]
	samples = []
	components=[]
	classes=[]
	for cl in range(2):
		comp.append(random.choices([0,1], [0.5,0.5], k=int(dataset_size/2)))
		for sample in comp[cl]:
			current_x_y = np.random.multivariate_normal([params[cl*4 + sample*2], params[cl*4 + sample*2 + 1]], [[1, 0], [0, 1]], 1)
			# print(current_x_y)
			samples.append(current_x_y[0].tolist())
			classes.append(cl)
			components.append(sample)

	return samples, classes

def gen_1d_object_dataset(params, output_folder, dataset_size=90, batch_size= 4):
	#generate images using blender and save them into a folder

	shutil.rmtree(output_folder)

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python3 simulation_utils/call_clevr.py -- --num_images %d --loc %f --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --render_quality_random 1" % (dataset_size/3, params, output_folder)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python3 simulation_utils/call_clevr.py -- --num_images %d --loc %f --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --render_quality_random 1" % (dataset_size/3, params, output_folder)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python3 simulation_utils/call_clevr.py -- --num_images %d --loc %f --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --render_quality_random 1" % (dataset_size/3, params, output_folder)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	



	# "cube": "SmoothCube_v2",
 #    "sphere": "Sphere",
 #    "cylinder": "SmoothCylinder"




	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_1dobject(root = root, train=2, mirror=False), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	return train_generator


def gen_quality_dataset(params, output_folder, dataset_size=90, batch_size= 4, writer=None):
	#generate images using blender and save them into a folder
	# shutil.rmtree(output_folder)
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	# for file_object in os.listdir(output_folder):
	#     file_object_path = os.path.join(output_folder, file_object)
	#     if os.path.isfile(file_object_path):
	#         os.unlink(file_object_path)
	#     else:
	#         shutil.rmtree(file_object_path)

	# if not os.path.isdir(output_folder):
	# 	os.makedirs(output_folder)

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --render_quality_prob %f" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --render_quality_prob %f" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --render_quality_prob %f" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	



	# "cube": "SmoothCube_v2",
 #    "sphere": "Sphere",
 #    "cylinder": "SmoothCylinder"




	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_1dobject(root = root, train=2, mirror=False, writer=writer), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	return train_generator

def gen_quality_dataset_budget(params, output_folder, dataset_size=90, time_budget=30, batch_size= 4):
	#generate images using blender and save them into a folder

	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	for file_object in os.listdir(output_folder):
	    file_object_path = os.path.join(output_folder, file_object)
	    if os.path.isfile(file_object_path):
	        os.unlink(file_object_path)
	    else:
	        shutil.rmtree(file_object_path)
	# logfile = open(subprocess.DEVNULL, 'w')

	start = time.time()
	status_dict = {'timeout':False}
	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --render_quality_prob %f" % (3000, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout = subprocess.DEVNULL)
	# output, error = process.communicate()

	timer = threading.Timer(int(time_budget/3), on_timeout, (process, status_dict))
	timer.start()
	process.wait()
	# in case we didn't hit timeout
	timer.cancel()


	status_dict['timeout'] = False
	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --render_quality_prob %f" % (3000, output_folder, params)
	# print(bashCommand)
	process1 = subprocess.Popen(bashCommand.split(), stdout = subprocess.DEVNULL)
	# output, error = process.communicate()
	timer = threading.Timer(int(time_budget/3), on_timeout, (process1, status_dict))
	timer.start()
	process1.wait()
	# in case we didn't hit timeout
	timer.cancel()

	status_dict['timeout'] = False
	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --render_quality_prob %f" % (3000, output_folder, params)
	# print(bashCommand)
	process2 = subprocess.Popen(bashCommand.split(), stdout = subprocess.DEVNULL)
	# output, error = process.communicate()
	timer = threading.Timer(int(time_budget/3), on_timeout, (process2, status_dict))
	timer.start()
	process2.wait()
	# in case we didn't hit timeout
	timer.cancel()



	# "cube": "SmoothCube_v2",
 #    "sphere": "Sphere",
 #    "cylinder": "SmoothCylinder"




	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_1dobject(root = root, train=2, mirror=False), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	return train_generator


def gen_quality_multiple(params, output_folder, dataset_size=90, batch_size= 4, writer=None, iter=0, last_k=0):
	#generate images using blender and save them into a folder
	# shutil.rmtree(output_folder)
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	for file_object in os.listdir(output_folder):
	    file_object_path = os.path.join(output_folder, file_object)
	    if os.path.isfile(file_object_path):
	        os.unlink(file_object_path)
	    else:
	        shutil.rmtree(file_object_path)

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr_multiple.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s" % (dataset_size/3, output_folder)
	bashCommand+=' --input_params'
	for in_par in range(len(params)):
		bashCommand+=' %f'%(params[in_par])
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr_multiple.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s" % (dataset_size/3, output_folder)
	bashCommand+=' --input_params'
	for in_par in range(len(params)):
		bashCommand+=' %f'%(params[in_par])
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr_multiple.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s" % (dataset_size/3, output_folder)
	bashCommand+=' --input_params'
	for in_par in range(len(params)):
		bashCommand+=' %f'%(params[in_par])
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	

	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_1dobject(root = root, train=2, mirror=False, writer=writer, iter=iter, last_k=last_k), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	return train_generator