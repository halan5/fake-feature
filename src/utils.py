
from PIL import Image
from datetime import datetime
from math import floor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import argparse
#import libmr
from math import ceil
import numpy as np
import os
import pandas
import random
import scipy.io as sio
#import sklearn.metrics.pairwise
import sys
import time
import torch
import torch.functional as func
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
#import torch.utils.model_zoo as model_zoo
import torchvision.datasets as dset
import torchvision.transforms as transforms
#import traceback
sys.path.append('../models/')

from models import *
hyper_para=None

def parse_args():
	parser = argparse.ArgumentParser()

	# optional arguments
	parser.add_argument("--batch_size"		  , default=64				  , type=int	  , help="batch size")
	parser.add_argument("--iterations"		  , default=500				 , type=float	, help="epoch number for training")
	parser.add_argument("dataset_name"		, type=str	  , help="dataset name")
	parser.add_argument("--model_border"		, default=None, 		type=int, help="point in the model for generating fake")
	parser.add_argument("--lr"				  , default=1e-4				, type=float	, help="learning rate")
	parser.add_argument("--no_closed"		   , default=6				  , type=int	  , help="number of known classes")
	parser.add_argument("--no_open"			 , default=4				   , type=int	  , help="number of unknown classes")
	parser.add_argument("--model_size"	   , default='VGG13'				, type=str , help="VGG-number")
	parser.add_argument("--default_para"	   , default=True				, type=str2bool , help="set false to override default parameters")
	
	return parser.parse_args()


def init_hyper_para(hyper_para_):
	global hyper_para
	hyper_para=hyper_para_

	
def handle_args(args):
	global hyper_para
	sys.path.append('../parameters/'+args.dataset_name+'/')
	if not args.default_para:
		hyper_para.lr 						= args.lr
		hyper_para.iterations				= args.iterations
		hyper_para.batch_size				= args.batch_size
		hyper_para.no_closed				= args.no_closed
		hyper_para.no_open					= args.no_open
		hyper_para.model_size				= args.model_size




def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


	
class ImageFolderDataset(Dataset):
	def __init__(self, root, transform):
		super(ImageFolderDataset, self).__init__()
		self.root=root
		self.transform=transform
		self.all_imgs = os.listdir(root)
	def __len__(self):
		return len(self.all_imgs)
	def __getitem__(self, idx):
		img_loc = os.path.join(self.root, self.all_imgs[idx])
		image = Image.open(img_loc).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, -1


def get_loaders():
	global hyper_para
	if hyper_para.dataset_name=='cifar10':
		return get_cifar10_loader()
	#elif hyper_para.dataset_name=='SVHN':
	#	return get_SVHN_loader()
	#etc
	else:
		print('ERROR: invalid dataset name')
		sys.exit()


def get_cifar10_loader():
	global hyper_para
	normalize = transforms.Normalize(
		mean=[0.4914, 0.4822, 0.4465],
		std=[0.2023, 0.1994, 0.2010],)
	
	# define transforms
	
	train_transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])
	train_dataset = dset.CIFAR10(
		root=hyper_para.dataset_path+'cifar10/', train=True,
		download=True, transform=train_transform,
	)
	test_transform = transforms.Compose([
	    transforms.ToTensor(),
	    normalize
	])
	kwn_indices=np.isin(train_dataset.targets, hyper_para.kwn)
	train_indices=np.arange(len(train_dataset.targets))
	np.random.shuffle(train_indices)
	train_indices=np.isin(np.arange(len(train_dataset.targets)), train_indices[:int(train_indices.size*0.8)])
	valid_indices=np.logical_not(train_indices)
	train_set=Subset(train_dataset, np.where(np.logical_and(kwn_indices, train_indices))[0])
	valid_dataset = dset.CIFAR10(
		root='../../datasets/cifar10/', train=True,
		download=True, transform=test_transform,
	)
	train_loader=DataLoader(train_set, batch_size=hyper_para.batch_size, shuffle=True)
	valid_set=Subset(valid_dataset, np.where(np.logical_and(kwn_indices, valid_indices))[0])	
	valid_loader=DataLoader(valid_set, batch_size=hyper_para.batch_size, shuffle=False)

	test_dataset=dset.CIFAR10(
		root='../../datasets/cifar10/', train=False,
		download=True, transform=test_transform,
	)
	kwn_indices=np.isin(test_dataset.targets, hyper_para.kwn)
	unk_indices=np.isin(test_dataset.targets, hyper_para.unk)
	kwn_testset=Subset(test_dataset, np.where(kwn_indices)[0])
	unk_testset=Subset(test_dataset, np.where(unk_indices)[0])
	test_loader_kwn=DataLoader(kwn_testset, batch_size=hyper_para.batch_size, shuffle=False)
	test_loader_unk=DataLoader(unk_testset, batch_size=hyper_para.batch_size, shuffle=False)
	return train_loader, valid_loader, test_loader_kwn, test_loader_unk


	
def save(variables, names, date):
	global hyper_para
	for (v, n) in zip(variables, names):
		sio.savemat('../save_folder/'+hyper_para.dataset_name+'/'+date+'/'+n+'.mat', {n: v})
def Permute_train(train_data, train_label):
	ind=np.random.permutation(train_label.size)
	train_label=train_label[ind]
	train_data=train_data[ind];
	return train_data, train_label


def anchor_loss(outputs, labels, anchor, weight=1):
	global hyper_para
	gt=anchor[labels]
	loss=torch.mean((outputs-gt)**2, dim=1)

	weights=torch.where(torch.max(torch.abs(gt), dim=1)[0]==0, torch.ones_like(loss), torch.ones_like(loss)*weight)

	loss=torch.mul(loss, weights)

	return torch.mean(loss)

def train_model(trainloader, kwn=None):
	global hyper_para
	if kwn is None:
		kwn=hyper_para.kwn

	anchor=torch.zeros([hyper_para.no_total+1, ceil(kwn.size/2)]).float()
	j=0
	k=1
	for i in kwn:
		if i==-1:
			continue
		anchor[i, floor(j/2)]=k
		k=-1*k
		j=j+1
	anchor=anchor.cuda()
	F_1 = VGG_part_one(True, hyper_para.model_border, hyper_para.feature_dim, hyper_para.image_channel, hyper_para.model_size)
	F_2 = VGG_part_two(True, hyper_para.model_border, int(kwn.size/2), hyper_para.feature_dim, hyper_para.model_size)
	if hyper_para.gpu:
		F_1.cuda()
		F_2.cuda()
	print(anchor)
	num_batches=len(trainloader)
	optimizer_1 = optim.Adam(F_1.parameters(), lr=hyper_para.lr, betas=hyper_para.betas)
	optimizer_2 = optim.Adam(F_2.parameters(), lr=hyper_para.lr, betas=hyper_para.betas)
	losses=torch.zeros(hyper_para.iterations*2).float()

	#fake_features=np.random.rand(train_label.size, 4096).astype(np.float32)
	for iteration in range(int(hyper_para.iterations)):
		t1 = time.time()
		epoch_loss=torch.tensor([0.]).cuda()
		bn=0
		for batch_idx, (inputs, labels) in enumerate(trainloader):
			t3=time.time()
			inputs = Variable(inputs).float()
			labels = Variable(labels).long()
			if hyper_para.gpu:
				inputs					 = inputs.cuda()
				labels 					 = labels.cuda()
			#t1=time.time()
			features = F_1(inputs)
			#t2=time.time()
			#print('time1: ' +str(t2-t1))
			loss=None
			optimizer_1.zero_grad()
			out=F_2(features)
			loss=anchor_loss(out, labels, anchor)
			optimizer_2.zero_grad()
			loss.backward()
			optimizer_1.step()
			optimizer_2.step()
			epoch_loss+=loss

		t2 = time.time()
		print('Epoch '+str(iteration)+', time spent: '+str(t2-t1))
		print('loss:'+str(epoch_loss/num_batches))
		losses[iteration]=(epoch_loss/num_batches).detach().cpu()
	F_1.eval()
	real_mid_features, train_label=run_model(F_1, trainloader)
	fake_mid_features=gan(real_mid_features, train_label)
	fake_labels=np.ones(fake_mid_features.shape[0])*hyper_para.no_total
	train_data=np.concatenate((real_mid_features, fake_mid_features))
	train_label=np.concatenate((train_label, fake_labels))
	num_batches=int(np.ceil(float(train_label.size)/float(hyper_para.batch_size)))
	for iteration in range(int(hyper_para.iterations)):
		t1 = time.time()
		train_data, train_label=Permute_train(train_data, train_label)
		epoch_loss=torch.tensor([0.]).cuda()
		
		for batch in range(num_batches):
			inputs, labels = Variable(torch.tensor(train_data[batch*hyper_para.batch_size:min((batch+1)*hyper_para.batch_size, train_label.size), :])), Variable(torch.tensor(train_label[batch*hyper_para.batch_size:min((batch+1)*hyper_para.batch_size, train_label.size)]))
			inputs = Variable(inputs).float()
			labels = Variable(labels).long()
			if hyper_para.gpu:
				inputs					 = inputs.cuda()
				labels 					 = labels.cuda()
			out=F_2(inputs)
			loss=anchor_loss(out, labels, anchor)
			optimizer_2.zero_grad()
			loss.backward()
			optimizer_2.step()
			epoch_loss+=loss
		t2 = time.time()
		print('Epoch '+str(iteration)+', time spent: '+str(t2-t1))
		print('loss:'+str(epoch_loss/num_batches))
		losses[hyper_para.iterations+iteration]=(epoch_loss/num_batches).detach().cpu()
		#losses can be saved and plotted if needed
	F_2.eval()
	return (F_1, F_2), losses, run_model(F_2, fake_mid_features)

def gan(train_data, train_label, conv=False):
	global hyper_para
	if conv:
		generator=Conv_Generator(train_data[1])
		discriminator=Conv_Discriminator(train_data[1])
	else:
		generator=Generator(train_data[1])
		discriminator=Discriminator(train_data[1])
	adversarial_loss = torch.nn.BCELoss()
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	Tensor = torch.cuda.FloatTensor if hyper_para.gpu else torch.FloatTensor
	num_batches=int(np.ceil(float(train_label.size)/float(64)))
	d_losses=torch.zeros(hyper_para.gan_iterations*num_batches).float()
	g_losses=torch.zeros(hyper_para.gan_iterations*num_batches).float()
	loss_ind=0
	for epoch in range(hyper_para.gan_iterations):
		for batch in range(num_batches):
			#time1=time.time()
			imgs, labels=train_data[batch*64:min((batch+1)*64, train_label.size), :], train_label[batch*64:min((batch+1)*64, train_label.size)]
			# Adversarial ground truths
			imgs=torch.tensor(imgs)
			valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).cuda()
			fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).cuda()
			# Configure input
			real_imgs = Variable(imgs.type(Tensor)).cuda()
			# -----------------
			#  Train Generator
			# -----------------
			optimizer_G.zero_grad()
			# Sample noise as generator input
			z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 128)))).cuda()
			# Generate a batch of images
			gen_imgs = generator(z)
			# Loss measures generator's ability to fool the discriminator
			g_loss = adversarial_loss(discriminator(gen_imgs), valid)
			g_loss.backward()
			optimizer_G.step()
			# ---------------------
			#  Train Discriminator
			# ---------------------
			optimizer_D.zero_grad()
			# Measure discriminator's ability to classify real from generated samples
			real_loss = adversarial_loss(discriminator(real_imgs), valid)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
			d_loss = (real_loss + fake_loss) / 2
			d_loss.backward()
			optimizer_D.step()
			#time2=time.time()
			#print('time: '+str(time2-time1))
			#print(
			#    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
			#    % (epoch, hyper_para.gan_iterations, batch, num_batches, d_loss.item(), g_loss.item())
			#)
			g_losses[loss_ind]=g_loss.item()
			d_losses[loss_ind]=d_loss.item()
			loss_ind=loss_ind+1
	generator.eval()
	X = Variable(Tensor(np.random.normal(0, 1, (4096, 128)))).cuda()
	num_batches=int(np.ceil(float(X.shape[0])/float(64)))
	
	inputs=Variable(torch.tensor(X[:64, :])).float()
	inputs=inputs.cuda()
	out=generator(inputs).detach().cpu()
	for batch in range(1, num_batches):
		inputs=Variable(torch.tensor(X[batch*64:min((batch+1)*64, X.shape[0]), :])).float()
		inputs=inputs.cuda()
		out=torch.cat((out, generator(inputs).detach().cpu()), dim=0)
	return out.cpu().detach().numpy()


def run_model(F, X):
	global hyper_para
	if type(F)!=tuple:
		F=(F,)
	if isinstance(X, DataLoader):
		out=None
		for batch_idx, (inputs, labels) in enumerate(X):
#			if inputs.dim()==5:
#				inputs=inputs.flatten(end_dim=1)
			inputs = Variable(inputs).float()
			labels = Variable(labels).long()
			if hyper_para.gpu:
				inputs					 = inputs.cuda()
			if len(F)==2:
				features=F[0](inputs)
				net=F[1]
			else:
				features=inputs
				net=F[0]
			out_=net(features).detach().cpu()
			if out is None:
				out=out_
				all_labels=labels.detach()
			else:
				out=torch.cat((out, out_), dim=0)
				all_labels=torch.cat((all_labels, labels.detach()))
		return out.numpy(), all_labels.numpy()
	else: #when inputs are tensors (fake features)
		num_batches=int(np.ceil(float(X.shape[0])/float(hyper_para.batch_size)))
		inputs=Variable(torch.tensor(X[:hyper_para.batch_size])).float()
		if hyper_para.gpu:
			inputs=inputs.cuda()
		
		out=None
		for batch in range(num_batches):
			inputs=Variable(torch.tensor(X[batch*hyper_para.batch_size:min((batch+1)*hyper_para.batch_size, X.shape[0])])).float()
			if hyper_para.gpu:
				inputs=inputs.cuda()
			features=inputs
			if len(F)==2:
				features=F[0](inputs)
				net=F[1]
			else:
				net=F[0]
			out_=net(features).detach().cpu()
			if out is None:
				out=out_
			else:
				out=torch.cat((out, out_), dim=0)
		return out.numpy()