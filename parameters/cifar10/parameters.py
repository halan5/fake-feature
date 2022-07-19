import random
import numpy as np
import torch
import math

class Hyperparameters():
	def __init__(self):
		#
		self.batch_size					 = 64
		self.iterations					 = 1
		self.gan_iterations				 = 1
		self.lr							 = 3e-4
		self.betas						 = (0.5, 0.99)
		self.model_size						 = 'VGG13'
		self.model_borders={
		'VGG11':7,
		'VGG13':14,
		'VGG16':17,
		'VGG19':20
		}
		self.model_border=self.model_borders[self.model_size]
		self.feature_dims={
		'VGG11':4096,
		'VGG13':512,
		'VGG16':512,
		'VGG19':512
		}
		self.feature_dim=self.feature_dims[self.model_size]

		self.image_channel				 = 3
		self.image_size					 = 32
		
		#
		self.gpu	 					 = True

		#
		self.dataset_name				 = 'cifar10'
		self.dataset_path 				 = '../../datasets/'
		
		#
		self.no_total					 = 10
		self.no_closed					 = 4
		self.no_open 					 = 6

		#
		self.kwn, self.unk				 = GetKwnUnkClasses(self.no_total, self.no_closed, self.no_open, 'random')

		self.normMean					 = [0.4914, 0.4822, 0.4465]
		self.normStd					 = [0.2023, 0.1994, 0.2010]

def GetKwnUnkClasses(no_total, no_closed, no_open, magic_word):

	if(magic_word=='sequential'):
		kwn = np.asarray(range(no_closed))
		unk = no_closed+np.asarray(range(np.min((no_open,no_total-no_closed))))
		print(kwn)
		print(unk)
	elif(magic_word=='random'):
		rand_id  = np.asarray(random.sample(range(no_total-no_closed),no_open))
		kwn = np.sort(np.asarray(random.sample(range(no_total),no_closed)))
		unk = np.asarray(np.where(np.in1d(np.asarray(range(no_total)),kwn)==False))[0,rand_id[0:no_open]]
		print(kwn)
		print(unk)
	elif(magic_word=='cifar_animals'):
		kwn=np.asarray([2, 3, 4, 5, 6, 7])
		unk=np.asarray([0, 1, 8, 9])
		print(kwn)
		print(unk)
	elif(magic_word=='all'):
		kwn=np.asarray([0,1,2,3,4,5,6,7,8,9])
		unk=np.asarray([])
	else:
		print('ERROR: known unknown split type not available')

	return kwn, unk

hyper_para = Hyperparameters()