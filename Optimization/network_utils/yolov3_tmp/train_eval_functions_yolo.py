import torch
import torchvision
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.optim import lr_scheduler
import time
from collections import OrderedDict
from scipy.optimize import fmin_ncg


import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import yolo_test  # import test.py to get mAP after each epoch
from models import *
from yolo_utils.datasets import *
from yolo_utils.utils import *

# mixed_precision = True
# try:  # Mixed precision training https://github.com/NVIDIA/apex
#     from apex import amp
# except:
#     mixed_precision = False  # not installed

# from apex import amp # Mixed precision training https://github.com/NVIDIA/apex
from torch.utils.tensorboard import SummaryWriter

SEED = 10
torch.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

def fmad(ys, xs, dxs):
	"""Forward-mode automatic differentiation."""
	v = [torch.zeros_like(pred_copy, requires_grad=True) for pred_copy in ys]
	# v = torch.zeros_like(ys, requires_grad=True)
	g = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True)
	# return torch.autograd.grad(g, v, grad_outputs=dxs)
	return [pred_copy for pred_copy in torch.autograd.grad(g, v, grad_outputs=dxs)]

def get_test_grad_loss_no_reg_val(model=None, validation_generator='1', device='1', criterion = nn.CrossEntropyLoss()):

	# if loss_type == 'normal_loss':
	#     op = self.grad_loss_no_reg_op
	# elif loss_type == 'adversarial_loss':
	#     op = self.grad_adversarial_loss_op
	# else:
	#     raise ValueError('Loss must be specified')
	test_grad_loss_no_reg_val = None
	model.zero_grad()
	with torch.set_grad_enabled(True):
		for i_iter, (imgs, targets, paths, _) in enumerate(validation_generator):
			# (imgs, targets, paths, _) = batch
			# Transfer to GPU
			print('device=',device)
			imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
			targets = targets.to(device)
			
			inf_out, train_out = model(imgs)
			# Compute loss
			loss, loss_items = compute_loss(train_out, targets, model)
			if not torch.isfinite(loss):
				print('WARNING: non-finite loss, ending training ', loss_items)
				return

			# Scale loss by nominal batch_size of 64
			loss *= imgs.shape[0] / 64.0
			loss.backward()
	# print("do we want to do normalization?")
	# test_grad_loss_no_reg_val = OrderedDict((name, param.grad.data/(i_iter+1)) for (name, param) in model.named_parameters())

	with torch.no_grad():
		# test_grad_loss_no_reg_val = [torch.flatten(param.grad) for param in model.parameters()]
		# test_grad_loss_no_reg_val = [param.grad.data/((i_iter+1)*len(local_labels)) for param in model.parameters()]
		test_grad_loss_no_reg_val = [param.grad.data/((i_iter+1)) for param in filter(lambda p: p.requires_grad, model.parameters())]

	model.zero_grad()
	return test_grad_loss_no_reg_val

def get_influence_on_test_loss(train_generator, model=None, inverse_hvp=None, inf_gen=None, device='1', criterion=nn.CrossEntropyLoss()):
	# print("Batch size must be 1 if you want individual influences")
	start_time = time.time()
	if inf_gen.batch_size != 1:
		raise ValueError("Batch size of inf_gen is not 1")
	num_to_add = len(inf_gen.dataset)
	predicted_loss_diffs = np.zeros([num_to_add])
	# predicted_loss_diffs = 0
	model.train()
	for counter, batch in enumerate(inf_gen):
		model.zero_grad()
		imgs, targets, paths, _ = batch
		# Transfer to GPU
		imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
		targets = targets.to(device)
		pred = model(imgs)
		# Compute loss
		loss, loss_items = compute_loss(pred, targets, model)
		if not torch.isfinite(loss):
			print('WARNING: non-finite loss, ending training ', loss_items)
			return

		# Scale loss by nominal batch_size of 64
		loss *= imgs.shape[0] / 64.0
		loss.backward()
		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in filter(lambda p: p.requires_grad, model.parameters())]
			# predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
			predicted_loss_diffs[counter] = np.dot(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)
			# predicted_loss_diffs += np.dot(np.concatenate([torch.flatten(t) for t in inverse_hvp]), np.concatenate(train_grad_loss_val))
	# print("do we want to do normalization?")
	duration = time.time() - start_time
	print('Multiplying by %s train examples took %s sec' % (num_to_add, duration))

	return predicted_loss_diffs

def get_total_influence_on_test_loss(train_generator, model=None, inverse_hvp=None, inf_gen=None, device='1', criterion=nn.CrossEntropyLoss()):
	# print("Batch size must be 1 if you want individual influences")
	start_time = time.time()
	if inf_gen.batch_size != 1:
		raise ValueError("Batch size of inf_gen is not 1")
	num_to_add = len(inf_gen.dataset)
	# predicted_loss_diffs = np.zeros([num_to_add])
	predicted_loss_diffs = 0
	model.train()
	for counter, batch in enumerate(inf_gen):
		model.zero_grad()
		imgs, targets, paths, _ = batch
		# Transfer to GPU
		imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
		targets = targets.to(device)
		pred = model(imgs)
		# Compute loss
		loss, loss_items = compute_loss(pred, targets, model)
		if not torch.isfinite(loss):
			print('WARNING: non-finite loss, ending training ', loss_items)
			return

		# Scale loss by nominal batch_size of 64
		loss *= imgs.shape[0] / 64.0
		loss.backward()
		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in filter(lambda p: p.requires_grad, model.parameters())]
			# print(np.linalg.norm(np.concatenate(train_grad_loss_val)))
		# predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
		# predicted_loss_diffs[counter] = np.dot(np.concatenate([torch.flatten(t) for t in inverse_hvp]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)
		predicted_loss_diffs += np.dot(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp]), np.concatenate(train_grad_loss_val))

	# print(np.linalg.norm(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp])))
	# print("do we want to do normalization?")
	duration = time.time() - start_time
	# print('Multiplying by %s train examples took %s sec' % (num_to_add, duration))

	return predicted_loss_diffs / len(train_generator.dataset)

def compute_inverse_hvp(model=None, train_generator=None, validation_generator=None, criterion=None, device='1', weight_decay=0.01,
	approx_type='cg', approx_params={'scale':25, 'recursion_depth':5000, 'damping':0, 'batch_size':1, 'num_samples':10}, force_refresh=True, test_description=None,
	X=None, Y=None, cg_max_iter=4, stoc_hessian=True, gauss_newton=0):

	test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(model, validation_generator, device=device, criterion=criterion)

	print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in test_grad_loss_no_reg_val])))

	start_time = time.time()
	if cg_max_iter == -1:
		return test_grad_loss_no_reg_val
	if cg_max_iter == -2:
		return [torch.ones_like(elmn) for elmn in test_grad_loss_no_reg_val]
	# approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
	# if os.path.exists(approx_filename) and force_refresh == False:
	#     inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
	#     print('Loaded inverse HVP from %s' % approx_filename)
	# else:
	#     inverse_hvp = self.get_inverse_hvp(
	#         test_grad_loss_no_reg_val,
	#         approx_type,
	#         approx_params)
	#     np.savez(approx_filename, inverse_hvp=inverse_hvp)
		# print('Saved inverse HVP to %s' % approx_filename)
	inverse_hvp = get_inverse_hvp(model, train_generator, validation_generator, test_grad_loss_no_reg_val, criterion, device, approx_type, approx_params, weight_decay, cg_max_iter, stoc_hessian, gauss_newton)
	duration = time.time() - start_time
	print('Inverse HVP took %s sec' % duration)
	return inverse_hvp

def get_inverse_hvp(model, train_generator, validation_generator, v, criterion, device, approx_type='lissa', approx_params=None, weight_decay =0.01, cg_max_iter=10, stoc_hessian=True, gauss_newton=0, verbose=False):
	assert approx_type in ['cg', 'lissa']
	if approx_type == 'lissa':
		return get_inverse_hvp_lissa(model, train_generator, v, criterion, **approx_params)
	elif approx_type == 'cg':
		if cg_max_iter == 0:
			return minibatch_hessian_vector_val(model, train_generator, v, stoc_hessian, gauss_newton, device)
		if cg_max_iter == -3:
			return [2*a - b for (a,b) in zip(v, minibatch_hessian_vector_val(model, train_generator, v, stoc_hessian, gauss_newton, device))]
		else:
			return get_inverse_hvp_cg(model, train_generator, validation_generator, v, weight_decay, cg_max_iter, stoc_hessian, gauss_newton, device, verbose)

def get_inverse_hvp_lissa(model, train_generator, v, criterion =None, batch_size=None, scale=10, damping=0.0, num_samples=1, recursion_depth=5000):
	"""
	This uses mini-batching; uncomment code for the single sample case.
	"""
	inverse_hvp = None
	print_iter = recursion_depth / 10

	for i in range(num_samples):
		# samples = np.random.choice(self.num_train_examples, size=recursion_depth)
		cur_estimate = v

		dataloader_iterator = iter(train_generator)
		for j in range(recursion_depth):
			try:
				local_batch, local_labels = next(dataloader_iterator)
			except:
				dataloader_iterator = iter(train_generator)
				local_batch, local_labels = next(dataloader_iterator)
			hessian_vector_val = hessian_vector_product(model, local_batch, local_labels, cur_estimate, criterion = criterion)
			with torch.no_grad():
				cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    

			# Update: v + (I - Hessian_at_x) * cur_estimate
			if (j % 10 == 0) or (j == recursion_depth - 1):
				print("Recursion at depth %s: norm is"%j)
				print(np.linalg.norm(np.concatenate([torch.flatten(x) for x in cur_estimate])))

		if inverse_hvp is None:
			inverse_hvp = [b/scale for b in cur_estimate]
		else:
			inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

	inverse_hvp = [a/num_samples for a in inverse_hvp]
	return inverse_hvp

def hessian_vector_product(model, local_batch, local_labels, v, gauss_newton):
	model.train()
	model.zero_grad()
	if not gauss_newton:
		# Run model
		# pred = model(torch.ones(local_batch.size()).cuda())
		pred = model(local_batch.cuda())

		# Compute loss
		# print(compute_loss(pred, local_labels, model))
		loss, loss_items = compute_loss(pred, local_labels, model)
		# loss=0
		# for pr in pred:
		# 	loss+=pr.sum()
		# loss = torch.norm(torch.cat([x.flatten() for x in pred]), p=2, dim=0)
		if not torch.isfinite(loss):
			print('WARNING: non-finite loss, ending training ', loss_items)
			return
		if torch.isnan(loss):
			print('WARNING: non-finite loss, ending training ', loss_items)
			return

		# Scale loss by nominal batch_size of 64
		# loss *= batch_size / 64
		loss *= local_batch.shape[0] / 64.0
		# print('loss is', loss)
		# Compute gradient
		# loss.backward()

		grads = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, model.parameters()), create_graph=True, retain_graph=True)
		# print(grads)
		# print('Norm of first order gradient: %s' % np.linalg.norm(np.concatenate([x.detach().flatten().cpu() for x in grads])))
		print('got the gradient')
		# z=0
		# for (param, grad) in zip(v, grads):
		# 	z+=(param*grad).sum()
		# print('z is ', z)
		# if torch.isnan(z):
		# 	print('WARNING: non-finite loss, ending training ', z)
		# 	return
		# grads_with_none = torch.autograd.grad(z, filter(lambda p: p.requires_grad, model.parameters()))
		# grads_with_none = torch.autograd.grad(z, model.parameters())
		# print('Norm of second order gradient: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in [grad_elem for grad_elem in grads_with_none]])))
		# grads_with_none = torch.autograd.grad(grads, filter(lambda p: p.requires_grad, model.parameters()), grad_outputs=v)
		grads_with_none = torch.autograd.grad(grads, filter(lambda p: p.requires_grad, model.parameters()), grad_outputs=v)
		# print('got the second order gradient')
		# return_grads = [grad_elem if grad_elem is not None \
		# else tf.zeros_like(x) \
		# for x, grad_elem in zip(v, grads_with_none)]
		return_grads = [grad_elem for grad_elem in grads_with_none]
	elif gauss_newton:
		predictions = model(local_batch)
		predictions_d = [pred_copy.detach().requires_grad_(True) for pred_copy in predictions]

		# Compute loss
		# print(compute_loss(pred, local_labels, model))
		loss, loss_items = compute_loss(predictions_d, local_labels, model)
		if not torch.isfinite(loss):
			print('WARNING: non-finite loss, ending training ', loss_items)
			return
		# Scale loss by nominal batch_size of 64
		# loss *= batch_size / 64
		loss *= local_batch.shape[0] / 64.0


		# compute J^T * z using FMAD (where z are the state variables)
		# (Jz,) = fmad(predictions, filter(lambda p: p.requires_grad, model.parameters()), v)  # equivalent but slower
		
		# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
		Jl = torch.autograd.grad(loss, predictions_d, create_graph=True)

		# compute loss Hessian (projected by Jz) using 2nd-order gradients
		(Hl_Jz,) = torch.autograd.grad(Jl, predictions_d, grad_outputs=v, retain_graph=True)
		print('Norm of second order gradient: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in Hl_Jz])))
		# compute J * (Hl_Jz) using RMAD (back-propagation).
		# note this is still missing the lambda * z term.
		# delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
		hessian_vector_val = torch.autograd.grad(predictions, filter(lambda p: p.requires_grad, model.parameters()), Hl_Jz, retain_graph=True)
		return_grads = [grad_elem for grad_elem in hessian_vector_val]

	print('Norm of second order gradient: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in return_grads])))
	return return_grads

def vec_to_list(model, v):
	return_list = []
	cur_pos = 0
	for p in filter(lambda p: p.requires_grad, model.parameters()):
		v_new = v[cur_pos : cur_pos+len(torch.flatten(p))].reshape(p.shape)
		return_list.append(torch.from_numpy(v_new).to(p.device))
		cur_pos += len(torch.flatten(p))

	assert cur_pos == len(v)
	# print([param for param in filter(lambda p: p.requires_grad, model.parameters())].size())
	return return_list


def get_fmin_loss_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device):
	def get_fmin_loss(x, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model,x), stoc_hessian, gauss_newton, device)
		print('Norm of hessian_vector_val: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in hessian_vector_val])))
		fmin_loss_val = 0.5 * np.dot(np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x, x) - np.dot(np.concatenate([l.flatten().cpu() for l in v]), x)
		print('fmin_loss value is:', fmin_loss_val)
		return fmin_loss_val
	return get_fmin_loss


def get_fmin_grad_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device):
	def get_fmin_grad(x, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, x), stoc_hessian, gauss_newton, device)
		
		return np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x - np.concatenate([l.flatten().cpu() for l in v])
	return get_fmin_grad

def get_fmin_hvp_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device):
	def get_fmin_hvp(x, p, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, p), stoc_hessian, gauss_newton, device)

		return np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + p
	return get_fmin_hvp

def get_cg_callback(model, train_generator, validation_generator, v, weight_decay, stoc_hessian, gauss_newton, device, verbose):
	fmin_loss_fn = get_fmin_loss_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device)
	
	def fmin_loss_split(x, model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, x), stoc_hessian, gauss_newton, device)

		return 0.5 * np.dot(np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x, x), - np.dot(np.concatenate([l.flatten().cpu() for l in v]), x)

	# def cg_callback(x, model, validation_generator, v):
	def cg_callback(x):
		# x is current params
		va = vec_to_list(model, x)
		idx_to_remove = 5

		model.zero_grad()
		criterion = nn.CrossEntropyLoss()
		model.train()
		with torch.set_grad_enabled(True):
			for counter, batch in enumerate(train_generator):
				imgs, targets, paths, _ = batch
				imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
				targets = targets.to(device)
				pred = model(imgs)
				# Compute loss
				loss, loss_items = compute_loss(pred, targets, model)
				if not torch.isfinite(loss):
					print('WARNING: non-finite loss, ending training ', loss_items)
					return

				loss *= imgs.shape[0] / 64.0
				loss.backward()
				break

		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in filter(lambda p: p.requires_grad, model.parameters())]
		predicted_loss_diff = np.dot(np.concatenate([torch.flatten(t).cpu() for t in va]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)

		# single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
		# train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
		# predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples
		print('Verbose is:', verbose)
		if verbose:
			print('Function value: %s' % fmin_loss_fn(x, model, train_generator, v))
			quad, lin = fmin_loss_split(x, model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device)
			print('Split function value: %s, %s' % (quad, lin))
			print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

	return cg_callback


def get_inverse_hvp_cg(model, train_generator, validation_generator, v, weight_decay, cg_max_iter, stoc_hessian, gauss_newton, device, verbose):
	fmin_loss_fn = get_fmin_loss_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device)
	fmin_grad_fn = get_fmin_grad_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device)
	cg_callback = get_cg_callback(model, train_generator, validation_generator, v, weight_decay, stoc_hessian, gauss_newton, device, verbose)
	get_fmin_hvp = get_fmin_hvp_fn(model, train_generator, v, weight_decay, stoc_hessian, gauss_newton, device)
	print('cg_max_iter is ', cg_max_iter)
	fmin_results = fmin_ncg(
		f=fmin_loss_fn,
		x0=np.concatenate([x.flatten().cpu() for x in v]),
		# avextol=0.001,
		fprime=fmin_grad_fn,
		fhess_p=get_fmin_hvp,
		callback=cg_callback,
		args=(model, train_generator, v),
		maxiter=cg_max_iter) 

	return vec_to_list(model, fmin_results)


def minibatch_hessian_vector_val(model, train_generator, v, stoc_hessian, gauss_newton, device, criterion = nn.CrossEntropyLoss()):
	damping=1e-2
	hessian_vector_val = None
	for i_iter, batch in enumerate(train_generator):
		imgs, targets, paths, _ = batch
		imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
		targets = targets.to(device)
		# Transfer to GPU
		# local_batch, local_labels = local_batch.to(device), local_labels.to(device)
		hessian_vector_val_temp = hessian_vector_product(model, imgs, targets, v, gauss_newton)

		if hessian_vector_val is None:
			hessian_vector_val = [b for b in hessian_vector_val_temp]
		else:
			hessian_vector_val = [a + b for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]

		###################################################
		########### DOING ONLY FOR ONE BATCH ##############
		###################################################
		if stoc_hessian:
			break

	hessian_vector_val = [a / float(i_iter+1) + damping * b for (a,b) in zip(hessian_vector_val, v)]

	return hessian_vector_val
