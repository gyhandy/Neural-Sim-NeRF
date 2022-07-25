import tensorflow as tf
import tensorboardX
import time
import csv
import sys
import os
import collections

# Import the event accumulator from Tensorboard. Location varies between Tensorflow versions. Try each known location until one works.
eventAccumulatorImported = False;
# TF version < 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.python.summary import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = 1.1.0
if (not eventAccumulatorImported):
	try:
		from tensorflow.tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version >= 1.3.0
if (not eventAccumulatorImported):
	try:
		from tensorboard.backend.event_processing import event_accumulator
		eventAccumulatorImported = True;
	except ImportError:
		eventAccumulatorImported = False;
# TF version = Unknown
if (not eventAccumulatorImported):
	raise ImportError('Could not locate and import Tensorflow event accumulator.')


class Timer(object):
	# Source: https://stackoverflow.com/a/5849861
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print('[%s]' % self.name)
			print('Elapsed: %s' % (time.time() - self.tstart))

# def exitWithUsage():
# 	print(' ');
# 	print('Usage:');
# 	print('   python readLogs.py <output-folder> <output-path-to-csv> <summaries>');
# 	print('Inputs:');
# 	print('   <input-path-to-logfile>  - Path to TensorFlow logfile.');
# 	print('   <output-folder>          - Path to output folder.');
# 	print('   <summaries>              - (Optional) Comma separated list of summaries to save in output-folder. Default: ' + ', '.join(summariesDefault));
# 	print(' ');
# 	sys.exit();

# if (len(sys.argv) < 3):
# 	exitWithUsage();

def get_opt_psi(inputLogFile):

	summaries = ['scalars'];

	print(' ');
	print('> Log file: ' + inputLogFile);
	print('Setting up event accumulator...');
	with Timer():
		ea = event_accumulator.EventAccumulator(inputLogFile,
	  	size_guidance={
	      	event_accumulator.COMPRESSED_HISTOGRAMS: 0, # 0 = grab all
	      	event_accumulator.IMAGES: 0,
	      	event_accumulator.AUDIO: 0,
	      	event_accumulator.SCALARS: 0,
	      	event_accumulator.HISTOGRAMS: 0,
		})

	print('Loading events from file*...');
	print('* This might take a while.');
	with Timer():
		ea.Reload() # loads events from file

	# print('Log summary:');
	tags = ea.Tags();
	# print(tags)
	for t in tags:
		tagSum = []
		if (isinstance(tags[t],collections.Sequence)):
			tagSum = str(len(tags[t])) + ' summaries';
		else:
			tagSum = str(tags[t]);
		# print('   ' + t + ': ' + tagSum);

	if ('scalars' in summaries):
		scalarTags = tags['scalars'];
		# print(scalarTags)
		with Timer():

			# Write headers to columns
			headers = ['wall_time','step'];
			for s in scalarTags:
				headers.append(s);
			if not scalarTags:
				return [], 0
			vals = ea.Scalars(scalarTags[0]);
			# print(vals)
			print(len(vals))
			# print(vals[5])
			best_mAP=0
			best_mean_policy=[]
			for i in range(len(vals)):
				v = vals[i];
				data = [v.wall_time, v.step];
				mean_policy=[]
				try:
					mean_pol_n=0
					for s in scalarTags:
						scalarTag = ea.Scalars(s);
						# print(scalarTag)
						try:
							S = scalarTag[i];
							data.append(S.value);
							if mean_pol_n < 44:
								mean_policy.append(S.value);
							mean_pol_n+=1
						except:
							pass
							# print(s, 'has', i, 'element missin')
					if i==0:
						best_mean_policy=mean_policy
						best_mAP=data[50]
					else:
						if data[50]>best_mAP:
							best_mAP=data[50]
							best_mean_policy=mean_policy
							# print(data)
							# print(data[0], data[45], data[46], data[50])
					# input('')
				except:
					pass
					print('Row %d is not complete'%i)

	return best_mean_policy, best_mAP