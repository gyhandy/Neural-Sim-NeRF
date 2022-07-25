import torch as t
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
import torch

def get_test_grad_loss_no_reg_val(model=None, validation_generator=None, device=None, criterion = nn.CrossEntropyLoss()):

	test_grad_loss_no_reg_val = None
	model.zero_grad()
	with torch.set_grad_enabled(True):
		for i_iter, batch in enumerate(validation_generator):
			local_batch, local_labels = batch
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
			output_batch = model(local_batch)
			loss = criterion(output_batch, local_labels)
			loss.backward()
	# print("do we want to do normalization?")
	# test_grad_loss_no_reg_val = OrderedDict((name, param.grad.data/(i_iter+1)) for (name, param) in model.named_parameters())

	with torch.no_grad():
		# test_grad_loss_no_reg_val = [torch.flatten(param.grad) for param in model.parameters()]
		# test_grad_loss_no_reg_val = [param.grad.data/((i_iter+1)*len(local_labels)) for param in model.parameters()]
		test_grad_loss_no_reg_val = [-param.grad.data/(float(i_iter+1)) for param in model.parameters()]

	model.zero_grad()
	return test_grad_loss_no_reg_val#return -J

	
class invhes_app(Optimizer):
	"""CurveBall optimizer with full Hessian instead of Gauss-Newton approximation"""
	def __init__(self, params, lr=None, momentum=None, auto_lambda=True, lambd=10.0,
			lambda_factor=0.999, lambda_low=0.5, lambda_high=1.5, lambda_interval=5):
		
		defaults = dict(lr=lr, momentum=momentum, auto_lambda=auto_lambda,
			lambd=lambd, lambda_factor=lambda_factor, lambda_low=lambda_low,
			lambda_high=lambda_high, lambda_interval=lambda_interval)
		super().__init__(params, defaults)

	def step(self, model, validation_generator, train_generator, criterion = nn.CrossEntropyLoss()):
		"""Performs a single optimization step"""

		# only support one parameter group
		if len(self.param_groups) != 1:
			raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
		group = self.param_groups[0]
		parameters = group['params']

		# initialize state to 0 if needed
		state = self.state
		for p in parameters:
			if p not in state:
				state[p] = {'z': t.zeros_like(p)}
		
		# linear list of state tensors z
		zs = [state[p]['z'] for p in parameters]
		
		# store global state (step count, lambda estimate) with first parameter
		global_state = state[parameters[0]]
		global_state.setdefault('count', 0)

		# get lambda estimate, or initial lambda (user hyper-parameter) if it's not set
		lambd = global_state.get('lambd', group['lambd'])
		

		#
		# compute CurveBall step (delta_zs)
		#
		#1
		#1.(a) J_val
		J = get_test_grad_loss_no_reg_val(model, validation_generator)

		
		#1.(b) J_train
		# TODO:
		# try loss.backward() instead of grad()?? (more efficient?)
		# hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, test_grad_loss_no_reg_val)
		Hz = None
		for i_iter, batch in enumerate(train_generator):
			local_batch, local_labels = batch
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

			# run forward pass
			model.zero_grad()
			output_batch = model(local_batch)
			loss = criterion(output_batch, local_labels)

			# compute gradient J, retaining the graph to allow 2nd-order gradients
			# J = grad(loss, parameters, create_graph=True)
			grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

			# compute H * z using 2nd-order gradients
			# Hz = grad(J, parameters, zs)
			hessian_vector_val = torch.autograd.grad(grads, model.parameters(), zs)

			if Hz is None:
				Hz = hessian_vector_val
			else:
				# for (z, dz) in zip(Hz, hessian_vector_val):
				# 	z.data.add_(dz)
				Hz = [a + b for (a,b) in zip(hessian_vector_val, Hz)]



		#2. delta zs
		# add Hessian, Jacobian and lambda terms to obtain delta_zs
		delta_zs = [j + hz/float(i_iter+1) + lambd * z for (j, hz, z) in zip(J, Hz, zs)]


		#######################################################
		# automatic hyper-parameters: momentum (rho) and learning rate (beta)
		#######################################################

		lr = group['lr']
		momentum = group['momentum']

		
		if momentum < 0 or lr < 0 or group['auto_lambda']:  # required by auto-lambda

			#### No computing Hdelta_zs
			H_delta_zs = None
			for i_iter, batch in enumerate(train_generator):
				local_batch, local_labels = batch
				local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

				# run forward pass
				model.zero_grad()
				output_batch = model(local_batch)
				loss = criterion(output_batch, local_labels)

				# compute gradient J, retaining the graph to allow 2nd-order gradients
				# J = grad(loss, parameters, create_graph=True)
				grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

				# compute H * z using 2nd-order gradients
				# Hz = grad(J, parameters, zs)
				hessian_vector_val = torch.autograd.grad(grads, model.parameters(), delta_zs)

				if H_delta_zs is None:
					H_delta_zs = hessian_vector_val
				else:
					H_delta_zs = [a + b for (a,b) in zip(hessian_vector_val, H_delta_zs)]

			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			z_vec = t.cat([z.flatten() for z in zs])
			dz_vec = t.cat([dz.flatten() for dz in delta_zs])

			a11 = lambd * (dz_vec * dz_vec).sum() + (dz_vec * t.cat([z.flatten() for z in H_delta_zs])).sum()
			a12 = lambd * (dz_vec * z_vec).sum() + (z_vec * t.cat([z.flatten() for z in H_delta_zs])).sum()
			a22 = lambd * (z_vec * z_vec).sum() + (z_vec * t.cat([z.flatten() for z in Hz])).sum()

			b1 = (t.cat([z.flatten() for z in J]) * dz_vec).sum()
			b2 = (t.cat([z.flatten() for z in J]) * z_vec).sum()

			# item() implicitly moves to the CPU
			A = t.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
			b = t.tensor([[b1.item()], [b2.item()]])
			auto_params = A.pinverse() @ b

			lr = auto_params[0].item()
			momentum = -auto_params[1].item()


		#######################################################
		# update parameters and state in-place: z = momentum * z + lr * delta_z; p = p + z
		#######################################################

		for (z, dz) in zip(zs, delta_zs):
			z.data.mul_(momentum).add_(-lr, dz)  # update state
			# p.data.add_(z)  # update parameter


		#######################################################
		# automatic lambda hyper-parameter (trust region adaptation)
		#######################################################

		if group['auto_lambda']:
			# only adapt once every few batches
			if global_state['count'] % group['lambda_interval'] == 0:
				with t.no_grad():
					# evaluate the loss with the updated parameters
					new_loss = loss_fn(model_fn())
					
					# objective function change predicted by quadratic fit
					quadratic_change = -0.5 * (auto_params * b).sum()

					# ratio between predicted and actual change
					ratio = (new_loss - loss) / quadratic_change

					# increase or decrease lambda based on ratio
					factor = group['lambda_factor'] ** group['lambda_interval']

					if ratio < group['lambda_low']: lambd /= factor
					if ratio > group['lambda_high']: lambd *= factor
					
					global_state['lambd'] = lambd
			global_state['count'] += 1

		# return (loss, predictions)
		return zs

class invhes_gauss_newton(Optimizer):
	"""CurveBall optimizer with full Hessian instead of Gauss-Newton approximation"""
	def __init__(self, params, lr=None, momentum=None, auto_lambda=True, lambd=10.0,
			lambda_factor=0.999, lambda_low=0.5, lambda_high=1.5, lambda_interval=5):
		
		defaults = dict(lr=lr, momentum=momentum, auto_lambda=auto_lambda,
			lambd=lambd, lambda_factor=lambda_factor, lambda_low=lambda_low,
			lambda_high=lambda_high, lambda_interval=lambda_interval)
		super().__init__(params, defaults)

	def step(self, model, validation_generator, train_generator, criterion = nn.CrossEntropyLoss()):
		"""Performs a single optimization step"""

		# only support one parameter group
		if len(self.param_groups) != 1:
			raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
		group = self.param_groups[0]
		parameters = group['params']

		# initialize state to 0 if needed
		state = self.state
		for p in parameters:
			if p not in state:
				state[p] = {'z': t.zeros_like(p)}
		
		# linear list of state tensors z
		zs = [state[p]['z'] for p in parameters]
		
		# store global state (step count, lambda estimate) with first parameter
		global_state = state[parameters[0]]
		global_state.setdefault('count', 0)

		# get lambda estimate, or initial lambda (user hyper-parameter) if it's not set
		lambd = global_state.get('lambd', group['lambd'])
		

		#
		# compute CurveBall step (delta_zs)
		#
		#1
		#1.(a) J_val
		J = get_test_grad_loss_no_reg_val(model, validation_generator)

		
		#1.(b) J_train
		# TODO:
		# try loss.backward() instead of grad()?? (more efficient?)
		# hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, test_grad_loss_no_reg_val)
		for i_iter, batch in enumerate(train_generator):
			local_batch, local_labels = batch
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

			# run forward pass
			# run forward pass, cutting off gradient propagation between model and loss function for efficiency
			model.zero_grad()
			predictions = model(local_batch)
			predictions_d = predictions.detach().requires_grad_(True)
			loss = criterion(predictions_d, local_labels)


			# compute J^T * z using FMAD (where z are the state variables)
			(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower
			
			# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
			(Jl,) = grad(loss, predictions_d, create_graph=True)

			# compute loss Hessian (projected by Jz) using 2nd-order gradients
			(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

			# compute J * (Hl_Jz) using RMAD (back-propagation).
			# note this is still missing the lambda * z term.
			# delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
			hessian_vector_val = grad(predictions, model.parameters(), Hl_Jz, retain_graph=True)

			if i_iter == 0:
				delta_zs = hessian_vector_val
			else:
				for (dz, z) in zip(delta_zs, hessian_vector_val):
					dz.data.add_(z)
				# delta_zs = [a + b for (a,b) in zip(hessian_vector_val, delta_zs)]


		# add lambda * z and J term to the result, obtaining the final steps delta_zs

		for (z, dz, j) in zip(zs, delta_zs, J):
			dz.data.div_(float(i_iter+1))
			dz.data.add_(lambd, z)
			dz.data.add_(j)

		print('check how to normalize')


		#
		# automatic hyper-parameters: momentum (rho) and learning rate (beta)
		#

		lr = group['lr']
		momentum = group['momentum']

		if momentum < 0 or lr < 0 or group['auto_lambda']:  # required by auto-lambda

			a11,a12,a22=0.0,0.0,0.0

			for i_iter, batch in enumerate(train_generator):
				local_batch, local_labels = batch
				local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

				# run forward pass
				# run forward pass, cutting off gradient propagation between model and loss function for efficiency
				model.zero_grad()
				predictions = model(local_batch)
				predictions_d = predictions.detach().requires_grad_(True)
				loss = criterion(predictions_d, local_labels)

				#compute J^T * z using FMAD (where z are the state variables)
				(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower

				# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
				(Jl,) = grad(loss, predictions_d, create_graph=True)

				# compute loss Hessian (projected by Jz) using 2nd-order gradients
				(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

				# compute J^T * delta_zs
				(Jdeltaz,) = fmad(predictions, model.parameters(), delta_zs)  # equivalent but slower

				# project result by loss hessian (using 2nd-order gradients)
				(Hl_Jdeltaz,) = grad(Jl, predictions_d, grad_outputs=Jdeltaz)
				

				with torch.no_grad():
					a11 = (Jdeltaz * Hl_Jdeltaz).sum() + a11
					a12 = (Jz * Hl_Jdeltaz).sum() + a12
					a22 = (Jz * Hl_Jz).sum() + a22
			

			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			z_vec = t.cat([z.flatten() for z in zs])
			dz_vec = t.cat([dz.flatten() for dz in delta_zs])

			# a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
			# a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
			# a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()

			a11 = lambd * (dz_vec * dz_vec).sum() + a11
			a12 = lambd * (dz_vec * z_vec).sum() + a12
			a22 = lambd * (z_vec * z_vec).sum() + a22


			##THIS PART LEFT
			# b1 = (Jl_d * Jdeltaz).sum()
			# b2 = (Jl_d * Jz).sum()
			b1 = (t.cat([z.flatten() for z in J]) * dz_vec).sum()
			b2 = (t.cat([z.flatten() for z in J]) * z_vec).sum()

			# item() implicitly moves to the CPU
			A = t.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
			b = t.tensor([[b1.item()], [b2.item()]])
			auto_params = A.pinverse() @ b

			lr = auto_params[0].item()
			momentum = -auto_params[1].item()


		#
		# update parameters and state in-place: z = momentum * z + lr * delta_z; p = p + z
		#

		for (z, dz) in zip(zs, delta_zs):
			z.data.mul_(momentum).add_(-lr, dz)  # update state


		#
		# automatic lambda hyper-parameter (trust region adaptation)
		#

		if group['auto_lambda']:
			# only adapt once every few batches
			if global_state['count'] % group['lambda_interval'] == 0:
				with t.no_grad():
					# evaluate the loss with the updated parameters
					new_loss = loss_fn(model_fn())
					
					# objective function change predicted by quadratic fit
					quadratic_change = -0.5 * (auto_params * b).sum()

					# ratio between predicted and actual change
					ratio = (new_loss - loss) / quadratic_change

					# increase or decrease lambda based on ratio
					factor = group['lambda_factor'] ** group['lambda_interval']

					if ratio < group['lambda_low']: lambd /= factor
					if ratio > group['lambda_high']: lambd *= factor
					
					global_state['lambd'] = lambd
			global_state['count'] += 1

		# return (loss, predictions)
		return zs

def fmad(ys, xs, dxs):
	"""Forward-mode automatic differentiation."""
	v = t.zeros_like(ys, requires_grad=True)
	g = grad(ys, xs, grad_outputs=v, create_graph=True)
	return grad(g, v, grad_outputs=dxs)

class invhes_aimn(Optimizer):
	"""CurveBall optimizer with full Hessian instead of Gauss-Newton approximation"""
	def __init__(self, params, lr=None, momentum=None, auto_lambda=True, lambd=10.0,
			lambda_factor=0.999, lambda_low=0.5, lambda_high=1.5, lambda_interval=5):
		
		defaults = dict(lr=lr, momentum=momentum, auto_lambda=auto_lambda,
			lambd=lambd, lambda_factor=lambda_factor, lambda_low=lambda_low,
			lambda_high=lambda_high, lambda_interval=lambda_interval)
		super().__init__(params, defaults)

	def step(self, model, validation_generator, train_generator, criterion = nn.CrossEntropyLoss()):
		"""Performs a single optimization step"""

		# only support one parameter group
		if len(self.param_groups) != 1:
			raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
		group = self.param_groups[0]
		parameters = group['params']

		# initialize state to 0 if needed
		state = self.state
		for p in parameters:
			if p not in state:
				state[p] = {'z': t.zeros_like(p), 'u': t.zeros_like(p)}
		
		# linear list of state tensors z
		zs = [state[p]['z'] for p in parameters]
		
		# store global state (step count, lambda estimate) with first parameter
		global_state = state[parameters[0]]
		global_state.setdefault('count', 0)

		# get lambda estimate, or initial lambda (user hyper-parameter) if it's not set
		lambd = global_state.get('lambd', group['lambd'])
		

		#
		# compute CurveBall step (delta_zs)
		#
		#1
		#1.(a) J_val
		J = get_test_grad_loss_no_reg_val(model, validation_generator)

		
		#1.(b) J_train
		# TODO:
		# try loss.backward() instead of grad()?? (more efficient?)
		# hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, test_grad_loss_no_reg_val)
		for i_iter, batch in enumerate(train_generator):
			local_batch, local_labels = batch
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

			# run forward pass
			# run forward pass, cutting off gradient propagation between model and loss function for efficiency
			model.zero_grad()
			predictions = model(local_batch)
			predictions_d = predictions.detach().requires_grad_(True)
			loss = criterion(predictions_d, local_labels)


			# compute J^T * z using FMAD (where z are the state variables)
			(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower
			
			# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
			(Jl,) = grad(loss, predictions_d, create_graph=True)

			# compute loss Hessian (projected by Jz) using 2nd-order gradients
			(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

			# compute J * (Hl_Jz) using RMAD (back-propagation).
			# note this is still missing the lambda * z term.
			# delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
			hessian_vector_val = grad(predictions, model.parameters(), Hl_Jz, retain_graph=True)

			if i_iter == 0:
				delta_zs = hessian_vector_val
			else:
				for (dz, z) in zip(delta_zs, hessian_vector_val):
					dz.data.add_(z)
				# delta_zs = [a + b for (a,b) in zip(hessian_vector_val, delta_zs)]


		# add lambda * z and J term to the result, obtaining the final steps delta_zs

		for (z, dz, j) in zip(zs, delta_zs, J):
			dz.data.div_(float(i_iter+1))
			dz.data.add_(lambd, z)
			dz.data.add_(j)

		# print('check how to normalize')


		#
		# automatic hyper-parameters: momentum (rho) and learning rate (beta)
		#

		lr = group['lr']
		momentum = group['momentum']

		if momentum < 0 or lr < 0 or group['auto_lambda']:  # required by auto-lambda

			a11,a12,a22=0.0,0.0,0.0

			for i_iter, batch in enumerate(train_generator):
				local_batch, local_labels = batch
				local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

				# run forward pass
				# run forward pass, cutting off gradient propagation between model and loss function for efficiency
				model.zero_grad()
				predictions = model(local_batch)
				predictions_d = predictions.detach().requires_grad_(True)
				loss = criterion(predictions_d, local_labels)

				#compute J^T * z using FMAD (where z are the state variables)
				(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower

				# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
				(Jl,) = grad(loss, predictions_d, create_graph=True)

				# compute loss Hessian (projected by Jz) using 2nd-order gradients
				(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

				# compute J^T * delta_zs
				(Jdeltaz,) = fmad(predictions, model.parameters(), delta_zs)  # equivalent but slower

				# project result by loss hessian (using 2nd-order gradients)
				(Hl_Jdeltaz,) = grad(Jl, predictions_d, grad_outputs=Jdeltaz)
				

				with torch.no_grad():
					a11 = (Jdeltaz * Hl_Jdeltaz).sum() + a11
					a12 = (Jz * Hl_Jdeltaz).sum() + a12
					a22 = (Jz * Hl_Jz).sum() + a22
			

			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			z_vec = t.cat([z.flatten() for z in zs])
			dz_vec = t.cat([dz.flatten() for dz in delta_zs])

			# a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
			# a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
			# a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()

			a11 = lambd * (dz_vec * dz_vec).sum() + a11
			a12 = lambd * (dz_vec * z_vec).sum() + a12
			a22 = lambd * (z_vec * z_vec).sum() + a22


			##THIS PART LEFT
			# b1 = (Jl_d * Jdeltaz).sum()
			# b2 = (Jl_d * Jz).sum()
			b1 = (t.cat([z.flatten() for z in J]) * dz_vec).sum()
			b2 = (t.cat([z.flatten() for z in J]) * z_vec).sum()

			# item() implicitly moves to the CPU
			A = t.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
			b = t.tensor([[b1.item()], [b2.item()]])
			auto_params = A.pinverse() @ b

			lr = auto_params[0].item()
			momentum = -auto_params[1].item()


		#
		# update parameters and state in-place: z = momentum * z + lr * delta_z; p = p + z
		#

		for (z, dz) in zip(zs, delta_zs):
			z.data.mul_(momentum).add_(-lr, dz)  # update state


		#
		# automatic lambda hyper-parameter (trust region adaptation)
		#

		if group['auto_lambda']:
			# only adapt once every few batches
			if global_state['count'] % group['lambda_interval'] == 0:
				with t.no_grad():
					# evaluate the loss with the updated parameters
					new_loss = loss_fn(model_fn())
					
					# objective function change predicted by quadratic fit
					quadratic_change = -0.5 * (auto_params * b).sum()

					# ratio between predicted and actual change
					ratio = (new_loss - loss) / quadratic_change

					# increase or decrease lambda based on ratio
					factor = group['lambda_factor'] ** group['lambda_interval']

					if ratio < group['lambda_low']: lambd /= factor
					if ratio > group['lambda_high']: lambd *= factor
					
					global_state['lambd'] = lambd
			global_state['count'] += 1

		# return (loss, predictions)
		return zs

	def step_nn(self, model, new_train_generator, train_generator, criterion = nn.CrossEntropyLoss()):
		"""Performs a single optimization step"""

		# only support one parameter group
		if len(self.param_groups) != 1:
			raise ValueError('Since the hyper-parameters are set automatically, only one parameter group (with the same hyper-parameters) is supported.')
		group = self.param_groups[0]
		parameters = group['params']

		# initialize state to 0 if needed
		state = self.state
		for p in parameters:
			if p not in state:
				state[p] = {'z': t.zeros_like(p), 'u': t.zeros_like(p)}
		
		# linear list of state tensors z
		zs = [state[p]['u'] for p in parameters]
		
		# store global state (step count, lambda estimate) with first parameter
		global_state = state[parameters[0]]
		global_state.setdefault('count', 0)

		# get lambda estimate, or initial lambda (user hyper-parameter) if it's not set
		lambd = global_state.get('lambd', group['lambd'])
		

		#
		# compute CurveBall step (delta_zs)
		#
		#1
		#1.(a) J_val
		J = get_test_grad_loss_no_reg_val(model, new_train_generator)

		
		#1.(b) J_train
		# TODO:
		# try loss.backward() instead of grad()?? (more efficient?)
		# hessian_vector_val = minibatch_hessian_vector_val(model, train_generator, test_grad_loss_no_reg_val)
		for i_iter, batch in enumerate(train_generator):
			local_batch, local_labels = batch
			local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

			# run forward pass
			# run forward pass, cutting off gradient propagation between model and loss function for efficiency
			model.zero_grad()
			predictions = model(local_batch)
			predictions_d = predictions.detach().requires_grad_(True)
			loss = criterion(predictions_d, local_labels)


			# compute J^T * z using FMAD (where z are the state variables)
			(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower
			
			# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
			(Jl,) = grad(loss, predictions_d, create_graph=True)

			# compute loss Hessian (projected by Jz) using 2nd-order gradients
			(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

			# compute J * (Hl_Jz) using RMAD (back-propagation).
			# note this is still missing the lambda * z term.
			# delta_zs = grad(predictions, parameters, Hl_Jz + Jl_d, retain_graph=True)
			hessian_vector_val = grad(predictions, model.parameters(), Hl_Jz, retain_graph=True)

			if i_iter == 0:
				delta_zs = hessian_vector_val
			else:
				for (dz, z) in zip(delta_zs, hessian_vector_val):
					dz.data.add_(z)
				# delta_zs = [a + b for (a,b) in zip(hessian_vector_val, delta_zs)]


		# add lambda * z and J term to the result, obtaining the final steps delta_zs

		for (z, dz, j) in zip(zs, delta_zs, J):
			dz.data.div_(float(i_iter+1))
			dz.data.add_(lambd, z)
			dz.data.add_(j)

		print('check how to normalize')


		#
		# automatic hyper-parameters: momentum (rho) and learning rate (beta)
		#

		lr = group['lr']
		momentum = group['momentum']

		if momentum < 0 or lr < 0 or group['auto_lambda']:  # required by auto-lambda

			a11,a12,a22=0.0,0.0,0.0

			for i_iter, batch in enumerate(train_generator):
				local_batch, local_labels = batch
				local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

				# run forward pass
				# run forward pass, cutting off gradient propagation between model and loss function for efficiency
				model.zero_grad()
				predictions = model(local_batch)
				predictions_d = predictions.detach().requires_grad_(True)
				loss = criterion(predictions_d, local_labels)

				#compute J^T * z using FMAD (where z are the state variables)
				(Jz,) = fmad(predictions, model.parameters(), zs)  # equivalent but slower

				# compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
				(Jl,) = grad(loss, predictions_d, create_graph=True)

				# compute loss Hessian (projected by Jz) using 2nd-order gradients
				(Hl_Jz,) = grad(Jl, predictions_d, grad_outputs=Jz, retain_graph=True)

				# compute J^T * delta_zs
				(Jdeltaz,) = fmad(predictions, model.parameters(), delta_zs)  # equivalent but slower

				# project result by loss hessian (using 2nd-order gradients)
				(Hl_Jdeltaz,) = grad(Jl, predictions_d, grad_outputs=Jdeltaz)
				

				with torch.no_grad():
					a11 = (Jdeltaz * Hl_Jdeltaz).sum() + a11
					a12 = (Jz * Hl_Jdeltaz).sum() + a12
					a22 = (Jz * Hl_Jz).sum() + a22
			

			# solve 2x2 linear system: [rho, -beta]^T = [a11, a12; a12, a22]^-1 [b1, b2]^T.
			# accumulate components of dot-product from all parameters, by first aggregating them into a vector.
			z_vec = t.cat([z.flatten() for z in zs])
			dz_vec = t.cat([dz.flatten() for dz in delta_zs])

			# a11 = lambd * (dz_vec * dz_vec).sum() + (Jdeltaz * Hl_Jdeltaz).sum()
			# a12 = lambd * (dz_vec * z_vec).sum() + (Jz * Hl_Jdeltaz).sum()
			# a22 = lambd * (z_vec * z_vec).sum() + (Jz * Hl_Jz).sum()

			a11 = lambd * (dz_vec * dz_vec).sum() + a11
			a12 = lambd * (dz_vec * z_vec).sum() + a12
			a22 = lambd * (z_vec * z_vec).sum() + a22


			##THIS PART LEFT
			# b1 = (Jl_d * Jdeltaz).sum()
			# b2 = (Jl_d * Jz).sum()
			b1 = (t.cat([z.flatten() for z in J]) * dz_vec).sum()
			b2 = (t.cat([z.flatten() for z in J]) * z_vec).sum()

			# item() implicitly moves to the CPU
			A = t.tensor([[a11.item(), a12.item()], [a12.item(), a22.item()]])
			b = t.tensor([[b1.item()], [b2.item()]])
			auto_params = A.pinverse() @ b

			lr = auto_params[0].item()
			momentum = -auto_params[1].item()


		#
		# update parameters and state in-place: z = momentum * z + lr * delta_z; p = p + z
		#
		with torch.no_grad():
			for (z, dz) in zip(zs, delta_zs):
				z.data.mul_(momentum).add_(-lr, dz)  # update state


		#
		# automatic lambda hyper-parameter (trust region adaptation)
		#

		if group['auto_lambda']:
			# only adapt once every few batches
			if global_state['count'] % group['lambda_interval'] == 0:
				with t.no_grad():
					# evaluate the loss with the updated parameters
					new_loss = loss_fn(model_fn())
					
					# objective function change predicted by quadratic fit
					quadratic_change = -0.5 * (auto_params * b).sum()

					# ratio between predicted and actual change
					ratio = (new_loss - loss) / quadratic_change

					# increase or decrease lambda based on ratio
					factor = group['lambda_factor'] ** group['lambda_interval']

					if ratio < group['lambda_low']: lambd /= factor
					if ratio > group['lambda_high']: lambd *= factor
					
					global_state['lambd'] = lambd
			global_state['count'] += 1

		# return (loss, predictions)
		with torch.no_grad():
			for (z, dz) in zip(model.parameters(), zs):
				z.data.add_(-1, dz)  # update state

		return