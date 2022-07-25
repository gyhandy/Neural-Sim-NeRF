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

from network_utils.datasets import Dataset_1dobject, Dataset_segobject
import shutil
import os
import time
import subprocess 
import threading
import sys

# SEED = 10
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

def delta_function(z):
	if z==0:
		return 1
	else:
		return 0
	   
def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	x = x/(sum(x) + delta_function(sum(x)))
	return x # only difference

def softmax_orig(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	# print(e_x)
	return e_x / e_x.sum(axis=0) # only difference

def on_timeout(proc, status_dict):
	"""Kill process on timeout and note as status_dict['timeout']=True"""
	# a container used to pass status back to calling thread
	status_dict['timeout'] = True
	# print("timed out")
	proc.kill()


def gen_yolo_dataset_lts(params, location_choices_files_path_prefix, train_txt, dataset_size=100):

	location_choices=['blackbed', 'blackshelf', 'blackshelf_sunlight','darkbrowncirculartable', 'darkbrownrectangulartable', 'darkbrownrectangulartable_nolight', 'darkredcircularchair', 'lightbrowncirculartable', 'lightgreysofa_light']
	probabilities_location_choices=softmax_orig(np.asarray([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]]))
	# print('probabilities_location_choices', probabilities_location_choices)
	samples_n = np.random.multinomial(dataset_size, probabilities_location_choices)
	samples_n = samples_n.tolist()
	# print(train_txt)
	with open(train_txt,"w+") as f:
		for loc_idx in range(len(location_choices)):
			# print(loc_idx)
			loc_name=location_choices[loc_idx]
			loc_num_samples=samples_n[loc_idx]
			location_choice_file_path = location_choices_files_path_prefix + loc_name +'.txt'
			# print(location_choice_file_path)
			with open(location_choice_file_path) as lf:
				all_lines=lf.readlines()
				if len(all_lines) < loc_num_samples:
					# lines = all_lines
					lines = filter('', all_lines)
					lines = filter('\n', lines)
				else:
					lines = random.sample(all_lines,loc_num_samples)
				# print(lines)
				for line in lines:
					f.write(line[6:])
					# print(line)
	return

def gen_yolo_dataset_ours(params, location_choices_files_path_prefix, train_txt, dataset_size=100, equal_sampling=False):

	location_choices=['blackbed', 'blackshelf', 'blackshelf_sunlight','darkbrowncirculartable', 'darkbrownrectangulartable', 'darkbrownrectangulartable_nolight', 'darkredcircularchair', 'lightbrowncirculartable', 'lightgreysofa_light']
	# probabilities_location_choices=softmax(np.asarray([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]]))
	if equal_sampling:
		probabilities_location_choices=np.asarray([1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0])
	else:
		probabilities_location_choices=np.asarray([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]])
	# print('probabilities_location_choices', probabilities_location_choices)
	samples_n = np.random.multinomial(dataset_size, probabilities_location_choices)
	samples_n = samples_n.tolist()
	# print(samples_n)
	# print(train_txt)
	with open(train_txt,"w+") as f:
		for loc_idx in range(len(location_choices)):
			loc_name=location_choices[loc_idx]
			loc_num_samples=samples_n[loc_idx]
			location_choice_file_path = location_choices_files_path_prefix + loc_name +'.txt'
			# print(location_choice_file_path)
			with open(location_choice_file_path) as lf:
				all_lines=lf.readlines()
				if len(all_lines) < loc_num_samples:
					lines = all_lines
					# lines = filter('', all_lines)
					# lines = filter('\n', lines)
				else:
					lines = random.sample(all_lines,loc_num_samples)
				# print(lines)
				for line in lines:
					f.write(line[6:])
					# print(line)
		# print(f)
	return samples_n


def gen_yolo_dataset_lts_big(params, location_choices_files_path_prefix, location_choices, train_txt, data_location='..', dataset_size=100, equal_sampling=False):
	# probabilities_location_choices=softmax_orig(np.asarray([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]]))
	if equal_sampling:
		probabilities_location_choices=np.ones(len(location_choices))/float(len(location_choices))
	else:
		probabilities_location_choices=softmax_orig(np.asarray(params))
	# print('probabilities_location_choices', probabilities_location_choices)
	samples_n = np.random.multinomial(dataset_size, probabilities_location_choices)
	samples_n = samples_n.tolist()
	# print(train_txt)
	with open(train_txt,"w+") as f:
		for loc_idx in range(len(location_choices)):
			# print(loc_idx)
			loc_name=location_choices[loc_idx].split('_')
			loc_num_samples=samples_n[loc_idx]
			location_choice_file_path = location_choices_files_path_prefix + loc_name[0] + '_' + loc_name[1] + os.sep + 'test_list_000_' + '_'.join(loc_name[2:]) +'.txt'
			# print(location_choice_file_path)
			with open(location_choice_file_path) as lf:
				all_lines=lf.readlines()
				# print(all_lines, loc_num_samples)
				if len(all_lines) < loc_num_samples:
					lines = all_lines
					# lines = filter('', all_lines)
					# lines = filter('\n', lines)
				else:
					lines = random.sample(all_lines,loc_num_samples)
				# print(lines)
				for line in lines:
					line_name = line[9:]
					line_name = data_location + os.sep + line_name
					# print(line_name)
					# print(line)
					f.write(line_name)
					# print(line)
	return

def gen_yolo_dataset_ours_big(params, location_choices_files_path_prefix, location_choices, train_txt, data_location='..', dataset_size=100, equal_sampling=False):

	# probabilities_location_choices=softmax(np.asarray([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]]))
	if equal_sampling:
		probabilities_location_choices=np.ones(len(location_choices))/float(len(location_choices))
	else:
		probabilities_location_choices=np.asarray(params)
	# print('probabilities_location_choices', probabilities_location_choices)
	samples_n = np.random.multinomial(dataset_size, probabilities_location_choices)
	samples_n = samples_n.tolist()
	# print(samples_n)
	# print(train_txt)
	with open(train_txt,"w+") as f:
		for loc_idx in range(len(location_choices)):
			loc_name=location_choices[loc_idx].split('_')
			loc_num_samples=samples_n[loc_idx]
			location_choice_file_path = location_choices_files_path_prefix + loc_name[0] + '_' + loc_name[1] + os.sep + 'test_list_000_' + '_'.join(loc_name[2:]) +'.txt'
			# print(location_choice_file_path)
			with open(location_choice_file_path) as lf:
				all_lines=lf.readlines()
				if len(all_lines) < loc_num_samples:
					lines = all_lines
				else:
					lines = random.sample(all_lines,loc_num_samples)
				# print(lines)
				for line in lines:
					line_name = line[9:]
					line_name = data_location + os.sep + line_name
					# print(line_name)
					# print(line)
					f.write(line_name)
					# print(line)
		# print(f)
	return samples_n

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

def gen_seg_dataset(params, output_folder, dataset_size=10, batch_size= 1, shuffle=True, equal_sampling=False, writer=None):
	#generate images using blender and save them into a folder
	# shutil.rmtree(output_folder)
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)
	# print(output_folder)
	if len(params) > 1:
		params=softmax_orig(np.asarray(params))

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender -noaudio --background --python simulation_utils/mul_obj_call_clevr.py -- --num_images %d --output_folder %s --render_quality_prob %f --use_gpu 1" % (dataset_size, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_segobject(root = root, train=2, mirror=False, writer=writer,crop_size=(32, 32)), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
	return train_generator

def gen_seg_dataset_mul(params, output_folder, dataset_size=10, batch_size= 1, shuffle=True, equal_sampling=False, writer=None, iter=0):
	#generate images using blender and save them into a folder
	# shutil.rmtree(output_folder)
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)
	###################################################################
	##################     SAMPLING PROCESS   ##################
	###################################################################
	params_list=[]
	count_list=[]
	total_params_list=[]
	#########################   SAMPLES  ##############################
	quality_choices=[2,128,512]
	if equal_sampling:
		probabilities_quality_choices=np.ones(len(quality_choices))/float(len(quality_choices))
	else:
		# print('Do we want to do softmax here?')
		probabilities_quality_choices=softmax_orig(np.asarray([params[0], params[1], params[2]]))
	# print('probabilities_quality_choices is ', probabilities_quality_choices)
	samples_n = np.random.multinomial(dataset_size, probabilities_quality_choices).tolist()
	samples_list=[]
	for qu_op in range(len(quality_choices)):
		for pi in range(samples_n[qu_op]):
			samples_list.append(qu_op)
	params_list.append(samples_list)
	count_list.append(samples_n)
	total_params_list.append(0)
	# samples_n = quality_choices[samples_n.index(1)]
	# render_num_samples=samples_n
	###################################################################
	#######################   IMAGE_SIZE  #############################
	imagesize_choices=[[32,32],[128,128],[256,256]]
	if equal_sampling:
		probabilities_size_choices=np.ones(len(imagesize_choices))/float(len(imagesize_choices))
	else:
		# print(np.asarray([params[3], params[4], params[5]]))
		probabilities_size_choices=softmax_orig(np.asarray([params[3], params[4], params[5]]))
	# print('probabilities_size_choices is ', probabilities_size_choices)
	size_n = np.random.multinomial(dataset_size, probabilities_size_choices).tolist()
	size_list=[]
	for qu_op in range(len(imagesize_choices)):
		for pi in range(size_n[qu_op]):
			size_list.append(qu_op)
	params_list.append(size_list)
	count_list.append(size_n)
	total_params_list.append(len(quality_choices))
	# size_n = imagesize_choices[size_n.index(1)]
	# args.width, args.height=size_n[0], size_n[1]
	###################################################################
	##########################   BOUNCES  #############################
	bounces_choices=[8,128]
	if equal_sampling:
		probabilities_bounces_choices=np.ones(len(bounces_choices))/float(len(bounces_choices))
	else:
		probabilities_bounces_choices=softmax_orig(np.asarray([params[6], params[7]]))
	# print('probabilities_bounces_choices is ', probabilities_bounces_choices)
	bounces_n = np.random.multinomial(dataset_size, probabilities_bounces_choices).tolist()
	bounces_list=[]
	for qu_op in range(len(bounces_choices)):
		for pi in range(bounces_n[qu_op]):
			bounces_list.append(qu_op)
	params_list.append(bounces_list)
	count_list.append(bounces_n)
	total_params_list.append(len(quality_choices) + len(imagesize_choices))
	# bounces_n = bounces_choices[bounces_n.index(1)]
	# args.render_max_bounces=bounces_n
	###################################################################
	####################   LIGHTS STRENGTH  ###########################
	params[8]=max(0,params[8])
	params[9]=max(0,params[9])
	params[8]=min(5,params[8])
	params[9]=min(100,params[9])
	light_strength_1=np.random.normal(params[8], 0.05, dataset_size).tolist()# from [0, 100]
	light_strength_2=np.random.normal(params[9], 0.05, dataset_size).tolist()# from [0, 100]
	params_list.append(light_strength_1)
	params_list.append(light_strength_2)
	count_list.append(dataset_size)
	count_list.append(dataset_size)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices))
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1)
	###################################################################
	##########################   LOCATION  ############################
	params[10]=max(-10,params[10])
	params[11]=max(-10,params[11])
	params[12]=max(-10,params[12])
	params[10]=min(3.3,params[10])
	params[11]=min(3.3,params[11])
	params[12]=min(3.3,params[12])
	print('think of something smarter to avoid occlusion')
	if max(params[10] - params[11], params[11] - params[10])<1:
		params[10] = params[11]+2
	if max(params[10] - params[12], params[12] - params[10])<1:
		params[12] = params[10]+2
	if max(params[11] - params[12], params[12] - params[11])<1:
		params[11] = params[12]+2
	x_loc1=np.random.normal(params[10], 0.05, dataset_size).tolist()
	x_loc2=np.random.normal(params[11], 0.05, dataset_size).tolist()
	x_loc3=np.random.normal(params[12], 0.05, dataset_size).tolist()
	params_list.append(x_loc1)
	params_list.append(x_loc2)
	params_list.append(x_loc3)
	count_list.append(dataset_size)
	count_list.append(dataset_size)
	count_list.append(dataset_size)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1 + 1)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1 + 1 + 1)
	###################################################################
	###########################   MATERIAL  ###########################
	material_mapping=[("Rubber", "rubber"), ("MyMetal", "metal")]
	if equal_sampling:
		probabilities_material_choices=np.ones(len(material_mapping))/float(len(material_mapping))
	else:
		probabilities_material_choices=softmax_orig(np.asarray([params[13], params[14]]))
	materials_n_1 = np.random.multinomial(dataset_size, probabilities_material_choices).tolist()
	materials_list_1=[]
	for qu_op in range(len(material_mapping)):
		for pi in range(materials_n_1[qu_op]):
			materials_list_1.append(qu_op)
	params_list.append(materials_list_1)
	count_list.append(materials_n_1)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1 + 1 + 1 + 1)
	#
	if equal_sampling:
		probabilities_material_choices=np.ones(len(material_mapping))/float(len(material_mapping))
	else:
		probabilities_material_choices=softmax_orig(np.asarray([params[15], params[16]]))
	materials_n_2 = np.random.multinomial(dataset_size, probabilities_material_choices).tolist()
	materials_list_2=[]
	for qu_op in range(len(material_mapping)):
		for pi in range(materials_n_2[qu_op]):
			materials_list_2.append(qu_op)
	params_list.append(materials_list_2)
	count_list.append(materials_n_2)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1 + 1 + 1 + 1 + len(material_mapping))
	#
	if equal_sampling:
		probabilities_material_choices=np.ones(len(material_mapping))/float(len(material_mapping))
	else:
		probabilities_material_choices=softmax_orig(np.asarray([params[17], params[18]]))
	materials_n_3 = np.random.multinomial(dataset_size, probabilities_material_choices).tolist()
	materials_list_3=[]
	for qu_op in range(len(material_mapping)):
		for pi in range(materials_n_3[qu_op]):
			materials_list_3.append(qu_op)
	params_list.append(materials_list_3)
	count_list.append(materials_n_3)
	total_params_list.append(len(quality_choices) + len(imagesize_choices) + len(bounces_choices) + 1 + 1 + 1 + 1 + 1 + len(material_mapping) + len(material_mapping))
	# print()
	# print('Size, y_loc, light2,3 and color of objects is left')
	# print()
	# mat_name, mat_name_out = material_mapping[materials_n.index(1)]
	###################################################################
	# print(params_list)
	# bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender -noaudio --background --python simulation_utils/mul_obj_call_clevr_multiple.py -- --num_images %d --output_folder %s --use_gpu 1 --params_list '{}'" % (dataset_size, output_folder, params_list)
	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender -noaudio --background --python simulation_utils/mul_obj_call_clevr_multiple.py -- --num_images %d --output_folder %s --use_gpu 1" % (dataset_size, output_folder)
	bashCommand+=' --params_list'
	for in_par in range(len(params_list)):
		for b_par in range(dataset_size):
			bashCommand+=" %f"%(params_list[in_par][b_par])
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	#######################################################
	########## read all images from that folder ###########
	#######################################################
	root = output_folder + '/images'
	# root = 'images_val_1d'
	train_generator = data.DataLoader(Dataset_segobject(root = root, train=2, mirror=False, writer=writer,crop_size=(32, 32), iter=iter), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
	return train_generator, params_list, count_list, total_params_list

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

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --render_quality_prob %f --use_gpu 1" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --render_quality_prob %f --use_gpu 1" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --render_quality_prob %f --use_gpu 1" % (dataset_size/3, output_folder, params)
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


def gen_size_dataset(params, output_folder, dataset_size=90, batch_size= 4, writer=None):
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

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --r %f --use_gpu 1" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --r %f --use_gpu 1" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --r %f --use_gpu 1" % (dataset_size/3, output_folder, params)
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

def gen_mat_dataset(params, output_folder, dataset_size=90, batch_size= 4, writer=None):
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

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder, params)
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder, params)
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



def gen_mul_dataset(params, output_folder, dataset_size=90, batch_size= 4, writer=None):
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

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCube_v2 --obj_name_out cube --filename_prefix cube --output_folder %s --r %f --render_quality_prob %f --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder, params[1], params[0], params[2])
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name Sphere --obj_name_out sphere --filename_prefix sphere --output_folder %s --r %f --render_quality_prob %f --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder,  params[1], params[0], params[2])
	# print(bashCommand)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "../blender-2.79b-linux-glibc219-x86_64/blender --background --python simulation_utils/call_clevr.py -- --num_images %d --obj_name SmoothCylinder --obj_name_out cylinder --filename_prefix cylinder --output_folder %s --r %f --render_quality_prob %f --render_material_prob %f --use_gpu 1 --render_material_random 0" % (dataset_size/3, output_folder,  params[1], params[0], params[2])
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