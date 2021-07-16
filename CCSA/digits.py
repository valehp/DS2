import random
import os
import numpy   as np
import sys
import scipy
import scipy.io as sio
import PIL
import gzip
import bz2
import pickle

import torch
from torch.utils.data import Dataset, random_split

import torchvision
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ToNumpy:
	def __init__(self):
		pass

	def __call__(self, img):
		img = np.array(img)
		return img


# ---------------------------------------- MNIST DATASET ---------------------------------------- #

def get_mnist(path='./data/MNIST/', img_size=(16,16), train=True, n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
	])
	# ..... mnist is source => return dataset ..... #
	if not n_samples:
		dataset = MNIST(path, transform=t, train=train, download=True)
		x, y = [], []
		max_len = 2000 if(tname=='usps') else len(dataset)
		for i in range(max_len):
			img, label = dataset[i]
			x.append(img)
			y.append(label)
		return x, y

	# ..... mnist is target => return n random samples ..... #
	samples_path = "{}{}_to_MNIST_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		dataset = MNIST(path, transform=t, train=train, download=True)
		img_labels = {}
		for i in range(10): img_labels[i] = []
		for i, data in enumerate(dataset):
			img_labels[ int(data[1]) ].append(i)
		
		x, y = [], []
		for label, indexs in img_labels.items():
			random.shuffle(indexs)
			for i in range(n):
				j = indexs[i]
				x.append(dataset[j][0])
				y.append(dataset[j][1])


		x = np.array(x)
		y = np.array(y)
		x = np.squeeze(x, 1)

		sio.savemat(samples_path, { 'X': x, 'y': y })

		if x.shape[1] == 1: x = np.moveaxis(x, 1, -1)
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return x, y

	data = sio.loadmat(samples_path)
	y = data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
	return np.array(data["X"]), np.array(y)


# ---------------------------------------- SVHN DATASET ---------------------------------------- #

def get_svhn(path='./data/SVHN/', img_size=(16,16), split='train', n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
		ToNumpy(),
	])
	# ..... is source => return hole dataset ----- #
	if not n_samples:
		dataset = SVHN(path, transform=t, split=split, download=True)
		x, y = [], []
		for img, label in dataset:
			x.append(img)
			y.append(label)
		return x, y

	# ..... is target => return n random samples ..... #
	samples_path = "{}{}_to_SVHN_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		dataset = SVHN(path, transform=t, split=split, download=True)
		img_labels = {}
		for i in range(10): img_labels[i] = []
		for i, data in enumerate(dataset):
			img_labels[ int(data[1]) ].append(i)
		
		x, y = [], []
		for label, indexs in img_labels.items():
			random.shuffle(indexs)
			for i in range(n):
				j = indexs[i]
				x.append(dataset[j][0])
				y.append(dataset[j][1])


		x = np.array(x)
		y = np.array(y)
		x = np.squeeze(x, 1)

		sio.savemat(samples_path, { 'X': x, 'y': y })

		if x.shape[1] == 1: x = np.moveaxis(x, 1, -1)
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return x, y

	data = sio.loadmat(samples_path)
	y = data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
	return np.array(data["X"]), np.array(y)

# ---------------------------------------- USPS DATASET ---------------------------------------- #

class DATASET(Dataset):
	def __init__(self, x, y, transform=None):
		self.x = x
		self.y = y
		self.t = transform
	
	def __len__(self):
		return len(self.x)

	def __getitem__(self, i):
		x, y = self.x[i], self.y[i]
		if self.t: x = self.t(x)
		return x, y

def get_usps(path='./data/USPS/', img_size=(16,16), train=True, n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.ToTensor(),
		transforms.ToPILImage(),
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
		ToNumpy(),
	])
	# ..... is source => return hole dataset ----- #
	if not n_samples:
		with gzip.open(path+'usps_28x28.pkl', 'rb') as f:
			(x_train, y_train), (x_test, y_test) = pickle.load(f, encoding='bytes')
		if train:
			print("TRAIN USPS ", x_train.shape)
			dataset = DATASET(x_train, y_train, t)
			X, Y = [], []
			max_len = 1800 if(tname=='mnist') else len(x_train)
			for i in range(max_len):
				x, y = dataset[i]
				X.append(x)
				Y.append(y)
			return X, Y
		print("TEST USPS ", x_test.shape)
		dataset = DATASET(x_test, y_test, t)
		X, Y = [], []
		for img, label in dataset:
			X.append(img)
			Y.append(label)
		return X, Y

	# ..... is target => return n random samples ..... #
	samples_path = "{}{}_to_USPS_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		with gzip.open(path+'usps_28x28.pkl', 'rb') as f:
			(x_train, y_train), (_, _) = pickle.load(f, encoding='bytes')

		dataset = DATASET(x_train, y_train, t)
		X, Y = [], []
		for x, y in dataset:
			X.append(x)
			Y.append(y)

		img_labels = {}
		for i in range(10): img_labels[i] = []
		for i in range(len(Y)):
			img_labels[ int(Y[i]) ].append(i)
		
		x, y = [], []
		for label, indexs in img_labels.items():
			random.shuffle(indexs)
			for i in range(n):
				j = indexs[i]
				x.append( X[j] )
				y.append( Y[j] )

		x = np.array(x)
		y = np.array(y)
		x = np.squeeze(x, 1)

		sio.savemat(samples_path, { 'X': x, 'y': y })
		if x.shape[1] == 1: x = np.moveaxis(x, 1, -1)
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return x, y

	data = sio.loadmat(samples_path)
	y = data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
	return np.array(data["X"]), np.array(y)

  
# ======================================== OFFICE DOMAIN ========================================= #

 
# ---------------------------------------- AMAZON dataset ---------------------------------------- #

def get_amazon(path='./data/OFFICE/amazon/', img_size=(16,16), train=True, n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
		ToNumpy(),
	])

	
	if train: data_path = path + 'amazon_16x16train.mat' 
	else: data_path = path + 'amazon_16x16test.mat'
	if not os.path.exists(data_path):
		dataset = ImageFolder(path+'images/', transform=t)

		print(len(dataset))
		test_size  = int( 0.25*len(dataset) )
		train_size = len(dataset)-test_size
		print("train -> ", train_size, " - test_size -> ", test_size)
		train_set, test_set = random_split( dataset, [train_size, test_size] )
		x_train, y_train, x_test, y_test = [], [], [], []
		for x, y in train_set:
			x_train.append(x)
			y_train.append(y)

		for x, y in test_set:
			x_test.append(x)
			y_test.append(y)

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test  = np.array(x_test)
		y_test  = np.array(y_test)

		x_train = np.squeeze(x_train, 1)
		x_test  = np.squeeze(x_test, 1)

		sio.savemat( path+'amazon_16x16train.mat', {'X': x_train, 'y': y_train} )
		sio.savemat( path+'amazon_16x16test.mat', {'X': x_test, 'y': y_test} )
		if not n_samples or not train:
			if train: return x_train, y_train
			return x_test, y_test

	if not n_samples:
		if train: data = sio.loadmat( path+'amazon_16x16train.mat' )
		else: data = sio.loadmat(path+'amazon_16x16test.mat')
		y = data["y"]
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return data["X"], y

	data = sio.loadmat( path+'amazon_16x16train.mat' )
	x, y = data["X"], data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)

	samples_path = "{}{}_to_AMAZON_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		labels = dict(zip( np.unique(y), [ [] for i in range(len( np.unique(y) )) ] ))

	x_samples, y_samples =  [], []
	for i in range(len(x)):
		labels[ int(y[i]) ].append(i)

	for label, idx in labels.items():
		random.shuffle(idx)
		for i in range(n):
			x_samples.append( x[ idx[i] ] )
			y_samples.append( y[ idx[i] ] )

	x = np.array(x_samples)
	y = np.array(y_samples)

	if len(x.shape) == 4 and x.shape[1] == 1: x = np.squeeze(x, 1)
	return x,y



# ---------------------------------------- DSLR dataset ---------------------------------------- #

def get_dslr(path='./data/OFFICE/dslr/', img_size=(16,16), train=True, n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
		ToNumpy(),
	])

	
	if train: data_path = path + 'dslr_16x16train.mat' 
	else: data_path = path + 'dslr_16x16test.mat'
	if not os.path.exists(data_path):
		dataset = ImageFolder(path+'images/', transform=t)

		test_size  = int( 0.25*len(dataset) )
		train_size = len(dataset)-test_size
		train_set, test_set = random_split( dataset, [train_size, test_size] )
		x_train, y_train, x_test, y_test = [], [], [], []
		for x, y in train_set:
			x_train.append(x)
			y_train.append(y)

		for x, y in test_set:
			x_test.append(x)
			y_test.append(y)

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test  = np.array(x_test)
		y_test  = np.array(y_test)

		x_train = np.squeeze(x_train, 1)
		x_test  = np.squeeze(x_test, 1)

		sio.savemat( path+'dslr_16x16train.mat', {'X': x_train, 'y': y_train} )
		sio.savemat( path+'dslr_16x16test.mat', {'X': x_test, 'y': y_test} )
		if not n_samples or not train:
			if train: return x_train, y_train
			return x_test, y_test

	if not n_samples:
		if train: data = sio.loadmat( path+'dslr_16x16train.mat' )
		else: data = sio.loadmat(path+'dslr_16x16test.mat')
		y = data["y"]
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return data["X"], y

	data = sio.loadmat( path+'dslr_16x16train.mat' )
	x, y = data["X"], data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)

	samples_path = "{}{}_to_DSLR_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		labels = dict(zip( np.unique(y), [ [] for i in range(len( np.unique(y) )) ] ))

	x_samples, y_samples =  [], []
	for i in range(len(x)):
		labels[ int(y[i]) ].append(i)

	for label, idx in labels.items():
		random.shuffle(idx)
		for i in range(n):
			x_samples.append( x[ idx[i] ] )
			y_samples.append( y[ idx[i] ] )

	x = np.array(x_samples)
	y = np.array(y_samples)

	if len(x.shape) == 4 and x.shape[1] == 1: x = np.squeeze(x, 1)
	return x,y





# ---------------------------------------- WEBCAM dataset ---------------------------------------- #

def get_webcam(path='./data/OFFICE/webcam/', img_size=(16,16), train=True, n_samples=False, n=1, source_name='mnist', tname=''):
	# soyrce_name -> for samples_path 
	t = transforms.Compose([
		transforms.Resize(img_size),
		transforms.Grayscale(),
		transforms.ToTensor(),
		transforms.Normalize( (0.5, ), (0.5) ),
		ToNumpy(),
	])

	
	if train: data_path = path + 'webcam_16x16train.mat' 
	else: data_path = path + 'webcam_16x16test.mat'
	if not os.path.exists(data_path):
		dataset = ImageFolder(path+'images/', transform=t)

		test_size  = int( 0.25*len(dataset) )
		train_size = len(dataset)-test_size
		train_set, test_set = random_split( dataset, [train_size, test_size] )
		x_train, y_train, x_test, y_test = [], [], [], []
		for x, y in train_set:
			x_train.append(x)
			y_train.append(y)

		for x, y in test_set:
			x_test.append(x)
			y_test.append(y)

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test  = np.array(x_test)
		y_test  = np.array(y_test)

		x_train = np.squeeze(x_train, 1)
		x_test  = np.squeeze(x_test, 1)

		sio.savemat( path+'webcam_16x16train.mat', {'X': x_train, 'y': y_train} )
		sio.savemat( path+'webcam_16x16test.mat', {'X': x_test, 'y': y_test} )
		if not n_samples or not train:
			if train:
				if x_train.shape[1] == 1: x_train = np.moveaxis(x_train, 1, -1)
				if y_train.shape[0] == 1 and len(y_train.shape) > 1: y_train = np.squeeze(y_train, 0)
				return x_train, y_train
			if x_test.shape[1] == 1: x_test = np.moveaxis(x_test, 1, -1)
			if y_test.shape[0] == 1 and len(y_test.shape) > 1: y_test = np.squeeze(y_train, 0)
			return x_test, y_test

	if not n_samples:
		if train: data = sio.loadmat( path+'webcam_16x16train.mat' )
		else: data = sio.loadmat(path+'webcam_16x16test.mat')
		x, y = data["X"], data["y"]
		if x.shape[1] == 1: x = np.moveaxis(x, 1, -1)
		if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)
		return x, y

	data = sio.loadmat( path+'webcam_16x16train.mat' )
	x, y = data["X"], data["y"]
	if y.shape[0] == 1 and len(y.shape) > 1: y = np.squeeze(y, 0)

	samples_path = "{}{}_to_WEBCAM_{}shot_{}x{}.mat".format(path, source_name.upper(), n, img_size[0], img_size[1])
	if not os.path.exists(samples_path):
		labels = dict(zip( np.unique(y), [ [] for i in range(len( np.unique(y) )) ] ))

	x_samples, y_samples =  [], []
	for i in range(len(x)):
		labels[ int(y[i]) ].append(i)

	for label, idx in labels.items():
		random.shuffle(idx)
		for i in range(n):
			x_samples.append( x[ idx[i] ] )
			y_samples.append( y[ idx[i] ] )

	x = np.array(x_samples)
	y = np.array(y_samples)

	if len(x.shape) == 4 and x.shape[1] == 1: x = np.squeeze(x, 1)
	return x,y