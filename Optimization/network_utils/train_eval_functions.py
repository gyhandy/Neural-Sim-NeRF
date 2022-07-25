import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
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

SEED = 10
torch.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

def train_network(model=None, max_epochs=20, train_gen=None, criterion = nn.CrossEntropyLoss(), optimizer = None, device=None, weight_decay=0.01):
	# 4. training
	if optimizer is None:
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
	# for epoch in tqdm(range(max_epochs)):
	with torch.set_grad_enabled(True):
		for epoch in range(max_epochs):
			# Training
			running_loss = 0.0
			for i_iter, batch in enumerate(train_gen):
				local_batch, local_labels = batch
				# Transfer to GPU
				local_batch, local_labels = local_batch.to(device), local_labels.to(device)
				# print(local_batch.shape)

				# Model computations
				optimizer.zero_grad()
				output_batch = model(local_batch)
				loss = criterion(output_batch, local_labels)
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

			scheduler.step()
			# print(epoch, running_loss)
	return model

def train_unet(model=None, max_epochs=20, train_gen=None, criterion = None, optimizer = None, device=None, weight_decay=0.01):
	# 4. training
	optim = torch.optim.Adam(model.parameters())
	# for epoch in tqdm(range(max_epochs)):
	with torch.set_grad_enabled(True):
		for epoch in range(max_epochs):
			# Training
			for i_iter, batch in enumerate(train_gen):
				local_batch, local_labels = batch
				# Transfer to GPU
				local_batch, local_labels = local_batch.to(device), local_labels.to(device).long()
				# print(local_batch.shape)

				# Model computations
				optim.zero_grad()
				output_batch = model(local_batch)
				loss = F.cross_entropy(output_batch, local_labels)
				loss.backward()
				optim.step()

			# print(epoch, running_loss)
	return model

def get_score_unet(network=None, valid_gen=None, device=None):
	# Validation
	correct = 0
	total = 0
	total_loss=0
	with torch.set_grad_enabled(False):
		for i_iter, batch in enumerate(valid_gen):
			# if i_iter >= 2:
			# 	break
			local_batch, local_labels = batch
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device).long()

			# Model computations
			output_batch = network(local_batch)
			loss = F.cross_entropy(output_batch, local_labels)
			total_loss += loss
			# _, predicted = torch.max(output_batch.data, 1)
			# total += local_labels.size(0)
			# correct += (predicted == local_labels).sum().item()

	# return correct / total
	return total_loss / i_iter

def get_score(network=None, valid_gen=None, device=None):
	# Validation
	criterion = nn.CrossEntropyLoss()
	correct = 0
	total = 0
	total_loss=0
	with torch.set_grad_enabled(False):
		for i_iter, batch in enumerate(valid_gen):
			# if i_iter >= 2:
			# 	break
			local_batch, local_labels = batch
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			# Model computations
			output_batch = network(local_batch)
			loss = criterion(output_batch, local_labels)
			total_loss += loss
			_, predicted = torch.max(output_batch.data, 1)
			total += local_labels.size(0)
			correct += (predicted == local_labels).sum().item()

	return correct / total
	# return total_loss / i_iter


def re_init(net):

	# Before I can zip() layers and pruning masks I need to make sure they match
	# one-to-one by removing all the irrelevant modules:
	prunable_layers = filter(
		lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
			layer, nn.Linear), net.modules())

	for layer in prunable_layers:

		nn.init.kaiming_normal_(layer.weight)

	return net


def get_test_grad_loss_no_reg_val(model=None, validation_generator=None, device=None, criterion = nn.CrossEntropyLoss()):

	# if loss_type == 'normal_loss':
	#     op = self.grad_loss_no_reg_op
	# elif loss_type == 'adversarial_loss':
	#     op = self.grad_adversarial_loss_op
	# else:
	#     raise ValueError('Loss must be specified')
	test_grad_loss_no_reg_val = None
	model.zero_grad()
	with torch.set_grad_enabled(True):
		for i_iter, batch in enumerate(validation_generator):
			local_batch, local_labels = batch
			# Transfer to GPU
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
			
			output_batch = model(local_batch)
			loss = criterion(output_batch, local_labels)
			loss.backward()
	# print("do we want to do normalization?")
	# test_grad_loss_no_reg_val = OrderedDict((name, param.grad.data/(i_iter+1)) for (name, param) in model.named_parameters())

	with torch.no_grad():
		# test_grad_loss_no_reg_val = [torch.flatten(param.grad) for param in model.parameters()]
		# test_grad_loss_no_reg_val = [param.grad.data/((i_iter+1)*len(local_labels)) for param in model.parameters()]
		test_grad_loss_no_reg_val = [param.grad.data/((i_iter+1)) for param in model.parameters()]

	model.zero_grad()
	return test_grad_loss_no_reg_val

def get_influence_on_test_loss(train_generator, model=None, inverse_hvp=None, inf_gen=None, criterion=nn.CrossEntropyLoss()):
	# print("Batch size must be 1 if you want individual influences")
	start_time = time.time()
	if inf_gen.batch_size != 1:
		raise ValueError("Batch size of inf_gen is not 1")
	num_to_add = len(inf_gen.dataset)
	predicted_loss_diffs = np.zeros([num_to_add])
	# predicted_loss_diffs = 0

	for counter, batch in enumerate(inf_gen):
		model.zero_grad()
		local_batch, local_labels = batch
		local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
		output_batch = model(local_batch)
		loss = criterion(output_batch, local_labels)
		loss.backward()
		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in model.parameters()]
			# predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
			predicted_loss_diffs[counter] = np.dot(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)
			# predicted_loss_diffs += np.dot(np.concatenate([torch.flatten(t) for t in inverse_hvp]), np.concatenate(train_grad_loss_val))
	# print("do we want to do normalization?")
	duration = time.time() - start_time
	print('Multiplying by %s train examples took %s sec' % (num_to_add, duration))

	return predicted_loss_diffs

def get_total_influence_on_test_loss(train_generator, model=None, inverse_hvp=None, inf_gen=None, criterion=nn.CrossEntropyLoss()):
	# print("Batch size must be 1 if you want individual influences")
	start_time = time.time()
	if inf_gen.batch_size != 1:
		raise ValueError("Batch size of inf_gen is not 1")
	num_to_add = len(inf_gen.dataset)
	# predicted_loss_diffs = np.zeros([num_to_add])
	predicted_loss_diffs = 0

	for counter, batch in enumerate(inf_gen):
		model.zero_grad()
		local_batch, local_labels = batch
		local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
		output_batch = model(local_batch)
		loss = criterion(output_batch, local_labels)
		loss.backward()
		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in model.parameters()]
			# print(np.linalg.norm(np.concatenate(train_grad_loss_val)))
		# predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
		# predicted_loss_diffs[counter] = np.dot(np.concatenate([torch.flatten(t) for t in inverse_hvp]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)
		predicted_loss_diffs += np.dot(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp]), np.concatenate(train_grad_loss_val))

	# print(np.linalg.norm(np.concatenate([torch.flatten(t).cpu() for t in inverse_hvp])))
	# print("do we want to do normalization?")
	duration = time.time() - start_time
	# print('Multiplying by %s train examples took %s sec' % (num_to_add, duration))

	return predicted_loss_diffs / len(train_generator.dataset)

def compute_inverse_hvp(model=None, train_generator=None, validation_generator=None, criterion=None, device=None, weight_decay=0.01,
	approx_type='cg', approx_params={'scale':25, 'recursion_depth':5000, 'damping':0, 'batch_size':1, 'num_samples':10}, force_refresh=True, test_description=None,
	X=None, Y=None, cg_max_iter=10, stoc_hessian=True, gauss_newton=0):

	test_grad_loss_no_reg_val = get_test_grad_loss_no_reg_val(model, validation_generator, device=device, criterion=criterion)

	print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate([x.flatten().cpu() for x in test_grad_loss_no_reg_val])))

	start_time = time.time()
	if cg_max_iter == -1:
		return test_grad_loss_no_reg_val
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
	inverse_hvp = get_inverse_hvp(model, train_generator, validation_generator, test_grad_loss_no_reg_val, criterion, approx_type, approx_params, weight_decay)
	duration = time.time() - start_time
	print('Inverse HVP took %s sec' % duration)
	return inverse_hvp

def get_inverse_hvp(model, train_generator, validation_generator, v, criterion, approx_type='lissa', approx_params=None, weight_decay =0.01, cg_max_iter=10, device='1', stoc_hessian=True,  gauss_newton=0, verbose=True):
	assert approx_type in ['cg', 'lissa']
	if approx_type == 'lissa':
		return get_inverse_hvp_lissa(model, train_generator, v, criterion, **approx_params)
	elif approx_type == 'cg':
		return get_inverse_hvp_cg(model, train_generator, validation_generator, v, weight_decay, verbose)

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

def hessian_vector_product(model, local_batch, local_labels, v, criterion = nn.CrossEntropyLoss()):
	model.zero_grad()
	local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
	output_batch = model(local_batch)
	loss = criterion(output_batch, local_labels)
	grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
	z=0
	for (param, grad) in zip(v, grads):
		z+=(param*grad).sum()
	grads_with_none = torch.autograd.grad(z, model.parameters())
	# return_grads = [grad_elem if grad_elem is not None \
	# else tf.zeros_like(x) \
	# for x, grad_elem in zip(v, grads_with_none)]
	return_grads = [grad_elem for grad_elem in grads_with_none]

	return return_grads

def vec_to_list(model, v):
	return_list = []
	cur_pos = 0
	for p in model.parameters():
		v_new = v[cur_pos : cur_pos+len(torch.flatten(p))].reshape(p.shape)
		return_list.append(torch.from_numpy(v_new).cuda())
		cur_pos += len(torch.flatten(p))

	assert cur_pos == len(v)
	
	return return_list


def get_fmin_loss_fn(model, train_generator, v, weight_decay):
	def get_fmin_loss(x, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model,x))

		return 0.5 * np.dot(np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x, x) - np.dot(np.concatenate([l.flatten().cpu() for l in v]), x)
	return get_fmin_loss


def get_fmin_grad_fn(model, train_generator, v, weight_decay):
	def get_fmin_grad(x, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, x))
		
		return np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x - np.concatenate([l.flatten().cpu() for l in v])
	return get_fmin_grad

def get_fmin_hvp_fn(model, train_generator, v, weight_decay):
	def get_fmin_hvp(x, p, model, train_generator, v):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, p))

		return np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + p
	return get_fmin_hvp

def get_cg_callback(model, train_generator, validation_generator, v, weight_decay, verbose):
	fmin_loss_fn = get_fmin_loss_fn(model, train_generator, v, weight_decay)
	
	def fmin_loss_split(x, model, train_generator, v, weight_decay):
		hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, vec_to_list(model, x))

		return 0.5 * np.dot(np.concatenate([torch.flatten(t).cpu() for t in hessian_vector_val])/weight_decay + x, x), - np.dot(np.concatenate([l.flatten().cpu() for l in v]), x)

	# def cg_callback(x, model, validation_generator, v):
	def cg_callback(x):
		# x is current params
		va = vec_to_list(model, x)
		idx_to_remove = 5

		model.zero_grad()
		criterion = nn.CrossEntropyLoss()
		with torch.set_grad_enabled(True):
			for counter, batch in enumerate(train_generator):
				local_batch, local_labels = batch
				local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
				output_batch = model(local_batch)
				loss = criterion(output_batch, local_labels)
				loss.backward()
				break

		with torch.no_grad():
			train_grad_loss_val = [torch.flatten(param.grad).cpu() for param in model.parameters()]
		predicted_loss_diff = np.dot(np.concatenate([torch.flatten(t).cpu() for t in va]), np.concatenate(train_grad_loss_val)) / len(train_generator.dataset)

		# single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
		# train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
		# predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples

		if verbose:
			print('Function value: %s' % fmin_loss_fn(x, model, train_generator, v))
			quad, lin = fmin_loss_split(x, model, train_generator, v, weight_decay)
			print('Split function value: %s, %s' % (quad, lin))
			print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

	return cg_callback


def get_inverse_hvp_cg(model, train_generator, validation_generator, v, weight_decay, verbose):
	fmin_loss_fn = get_fmin_loss_fn(model, train_generator, v, weight_decay)
	fmin_grad_fn = get_fmin_grad_fn(model, train_generator, v, weight_decay)
	cg_callback = get_cg_callback(model, train_generator, validation_generator, v, weight_decay, verbose)
	get_fmin_hvp = get_fmin_hvp_fn(model, train_generator, v, weight_decay)

	fmin_results = fmin_ncg(
		f=fmin_loss_fn,
		x0=np.concatenate([x.flatten().cpu() for x in v]),
		fprime=fmin_grad_fn,
		fhess_p=get_fmin_hvp,
		callback=cg_callback,
		args=(model, train_generator, v),
		avextol=1e-8,
		maxiter=100) 

	return vec_to_list(model, fmin_results)


def minibatch_hessian_vector_val(model, train_generator, v, criterion = nn.CrossEntropyLoss()):
	damping=1e-2
	hessian_vector_val = None
	for i_iter, batch in enumerate(train_generator):
		local_batch, local_labels = batch
		# Transfer to GPU
		# local_batch, local_labels = local_batch.to(device), local_labels.to(device)
		hessian_vector_val_temp = hessian_vector_product(model, local_batch, local_labels, v, criterion = criterion)

		if hessian_vector_val is None:
			hessian_vector_val = [b for b in hessian_vector_val_temp]
		else:
			hessian_vector_val = [a + b for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]

	hessian_vector_val = [a / float(i_iter+1) + damping * b for (a,b) in zip(hessian_vector_val, v)]

	return hessian_vector_val

def find_eigvals_of_hessian(self, num_iter=100, num_prints=10):

        # Setup
        print_iterations = num_iter / num_prints
        feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, 0)

        # Initialize starting vector
        grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=feed_dict)
        initial_v = []

        for a in grad_loss_val:
            initial_v.append(np.random.random(a.shape))        
        initial_v, _ = normalize_vector(initial_v)

        # Do power iteration to find largest eigenvalue
        print('Starting power iteration to find largest eigenvalue...')

        largest_eig = norm_val
        print('Largest eigenvalue is %s' % largest_eig)

        # Do power iteration to find smallest eigenvalue
        print('Starting power iteration to find smallest eigenvalue...')
        cur_estimate = initial_v
        
        for i in range(num_iter):
            cur_estimate, norm_val = normalize_vector(cur_estimate)
            hessian_vector_val = self.minibatch_hessian_vector_val(cur_estimate)
            new_cur_estimate = [a - largest_eig * b for (a,b) in zip(hessian_vector_val, cur_estimate)]

            if i % print_iterations == 0:
                print(-norm_val + largest_eig)
                dotp = np.dot(np.concatenate(new_cur_estimate), np.concatenate(cur_estimate))
                print("dot: %s" % dotp)
            cur_estimate = new_cur_estimate

        smallest_eig = -norm_val + largest_eig
        assert dotp < 0, "Eigenvalue calc failed to find largest eigenvalue"

        print('Largest eigenvalue is %s' % largest_eig)
        print('Smallest eigenvalue is %s' % smallest_eig)
        return largest_eig, smallest_eig
