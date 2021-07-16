# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader, Dataset #, random_split
import os
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
import gzip
import bz2 as bzip2
import pickle as pic
import _pickle as pickle
import numpy as np
import random
import time
import scipy.io as sio


class ToNumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)

# ------------------------------- custom datasets ------------------------------------- #

class OFFICEDataset(Dataset):
	def __init__(self, x, y, transform=None):
		self.x = x
		self.y = torch.tensor(y)
		self.t = transform

	def __len__(self):
		return len(self.x)

	def __getitem__(self, i):
		img, label = self.x[i], self.y[i]
		if self.t: img = self.t(img)
		return img, label


class DCD_dataset(Dataset):
	def __init__(self, x1, y1, x2, y2, truth, fake):
		self.x1 = x1
		self.y1 = torch.LongTensor(y1)
		self.x2 = x2
		self.y2 = torch.LongTensor(y2)
		self.truth = torch.LongTensor(truth)
		self.fake  = torch.LongTensor(fake)
		if len(self.truth.size()) == 2: self.truth = self.truth.squeeze(1)
		if len(self.fake.size()) == 2: self.fake = self.fake.squeeze(1)

	def __len__(self):
		return len(self.truth)

	def __getitem__(self, i):
		return (self.x1[i], self.y1[i], self.x2[i], self.y2[i], self.truth[i], self.fake[i])


class SVHNDataset(Dataset): # for SVHN only
	def __init__(self, data, labels, transform=None):
		self.data = data
		self.L = True
		self.labels = torch.from_numpy(np.array(labels))    
		if len(self.labels.size()) > 1:
			if self.labels.size()[0] == 1: self.labels = self.labels.squeeze(0)
			if self.labels.size()[1] == 1: self.labels = self.labels.squeeze(1)
		self.trans = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		img, label = self.data[i], self.labels[i]
		if self.trans: img = self.trans(img)
		return img, label

class MNISTDataset(Dataset): # for MNIST and USPS
	def __init__(self, data, labels, transform=None):
		self.data = data
		self.L = True
		self.labels = torch.from_numpy(np.array(labels))  
		self.labels = torch.squeeze(self.labels, dim=0)  
		self.trans = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		img, label = self.data[i], self.labels[i]
		if self.trans: img = self.trans( np.expand_dims(img, axis=-1) )
		return img, label

class USPSDataset(Dataset): # for MNIST and USPS
	def __init__(self, data, labels, transform=None):
		self.data = data
		self.L = True
		self.labels = torch.from_numpy(np.array(labels))
		self.trans = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		img, label = np.squeeze(self.data[i], 0), self.labels[i]
		if self.trans: img = self.trans( img )
		return img, label


# ------------------------------- SVHN dataset ------------------------------------- #

def get_svhn(path='./data/SVHN', split='train', transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="mnist"):
	if source or split == 'test':
		dataloader = DataLoader( SVHN(path, split=split, transform=transform, download=True), batch_size=batch_size, shuffle=True )
		return dataloader

	labels_path = "{}/{}_to_SVHN_{}shots_FADA.pkl".format(path, source_name.upper(), n)
	if not os.path.exists(labels_path):
		t_aux = transforms.Compose([ToNumpy()])
		file = bzip2.BZ2File(labels_path, 'wb')
		labels = {}
		for i in range(10): labels[i] = []
		data = SVHN(path, split=split, transform=t_aux, download=True)
		for i in range(len(data)):
			labels[ data[i][1] ].append(i)

		x, y = [], []
		for label, index in labels.items():
			t = []
			for i in index:
				t.append( (data[i][0], data[i][1]) )
			random.shuffle(t)
			for i in range(n):
				x.append(t[i][0])
				y.append(t[i][1])

		pickle.dump( (x,y), file )
		file.close()
		ds = SVHNDataset(x, y, transform=transform)
		x, y = [], []
		for img, label in ds:
			x.append(img)
			y.append(label)
		return x, y

	file = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	ds = SVHNDataset(x, y, transform=transform)
	x, y = [], []
	for img, label in ds:
		x.append(img)
		y.append(label)
	return x, y



# ------------------------------- MNIST dataset ------------------------------------- #

def get_mnist(path='./data/MNIST', train=True, transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="usps"):
	if source or not train:
		dataloader = DataLoader( MNIST(path, train=train, transform=transform, download=True), batch_size=batch_size, shuffle=True )
		return dataloader
	
	labels_path = "{}/{}_to_MNIST_{}shots_FADA.pkl".format(path, source_name.upper(), n)

	if not os.path.exists(labels_path):
		t_aux = transforms.Compose([ToNumpy()])
		file = bzip2.BZ2File(labels_path, 'wb')
		labels = {}
		for i in range(10): labels[i] = []
		data = MNIST(path, train=train, transform=t_aux, download=True)
		if source_name == 'usps': max_len = 2000
		else: max_len = len(data)

		for i in range(max_len):
			labels[ data[i][1] ].append(i)

		x, y = [], []
		for label, index in labels.items():
			t = []
			for i in index:
				t.append( (data[i][0], data[i][1]) )
			random.shuffle(t)
			for i in range(n):
				x.append(t[i][0])
				y.append( np.array(t[i][1]) )

		pickle.dump( (x,y), file )
		file.close()
		ds = MNISTDataset(x, y, transform=transform)
		x, y = [], []
		for img, label in ds:
			x.append(img)
			y.append(label)
		return x, y

	file = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	ds = MNISTDataset(x, y, transform=transform)
	x, y = [], []
	for img, label in ds:
		x.append(img)
		y.append(label)
	return x, y


# ------------------------------- USPS dataset ------------------------------------- #

def get_usps(path='./data/USPS', train=True, transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="mnist"):
	if source or not train:
		with gzip.open(path+'/usps_28x28.pkl', 'rb') as f:
			(x_train, y_train), (x_test, y_test) = pic.load(f, encoding='bytes')
		if train: return DataLoader( USPSDataset(x_train, y_train, transform=transform), batch_size=batch_size, shuffle=True )
		else: return DataLoader( USPSDataset(x_test, y_test, transform=transform), batch_size=batch_size, shuffle=True )

	labels_path = "{}/{}_to_USPS_{}shots_FADA.pkl".format(path, source_name.upper(), n)
	if not os.path.exists(labels_path):
		t_aux = transforms.Compose([ToNumpy()])
		file = bzip2.BZ2File(labels_path, 'wb')
		labels = {}
		for i in range(10): labels[i] = []

		with gzip.open(path+'/usps_28x28.pkl', 'rb') as f:
			(x_train, y_train), (_, _) = pickle.load(f, encoding='bytes')

		if source_name == 'mnist': max_len = 1800
		else: max_len = len(y_train)

		for i in range(max_len):
			labels[ y_train[i] ].append(i)
	
		x, y = [], []
		for label, index in labels.items():
			t = []
			for i in index:
				t.append( (x_train[i], y_train[i]) )
			random.shuffle(t)
			for i in range(n):
				x.append(t[i][0])
				y.append(t[i][1])

		pickle.dump( (x,y), file )
		file.close()
		ds = USPSDataset(x, y, transform=transform)
		x, y = [], []
		for i in range(len(ds)):
			x.append(ds[i][0])
			y.append(ds[i][1])
		return x, y

	file  = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	ds = USPSDataset(x, y, transform=transform)
	x, y = [], []
	for img, label in ds:
		x.append(img)
		y.append(label)
	return x, y



# ------------------------------- AMAZON dataset ------------------------------------- #

def get_amazon(path='./data/AMAZON/', train=True, transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="mnist"):
	if source or not train:
		data = sio.loadmat(path+'/amazon.mat')
		if train: x, y = data["X"], data["y"]
		else: x, y = data["X_test"], data["y_test"]
		data.clear()
		return DataLoader( OFFICEDataset(x, y, transform), batch_size=batch_size, shuffle=True )

	labels_path = "{}{}_to_AMAZON_{}shots_FADA.pkl".format(path, source_name.upper(), n)
	if not os.path.exists(labels_path):
		file = bzip2.BZ2File(labels_path, 'wb')
		data = sio.loadmat(path+'/amazon.mat')
		x_train, y_train = data["X"], data["y"]
		data.clear()
		data = OFFICEDataset(x_train, y_train, transform)

		labels = {}
		for i in range(31): labels[i] = []
		for i in range(len(data)):
			labels[ data[i][1].item() ].append(i)

		x, y = [], []
		for label, index in labels.items():
			random.shuffle(index)
			for i in range(n):
				j = index[i]
				x.append( data[j][0] )
				y.append( data[j][1] )

		pickle.dump( (x,y), file )
		file.close()
		return x, y

	file  = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	return x, y


# ------------------------------- DSLR dataset ------------------------------------- #


def get_dslr(path='./data/DSLR/', train=True, transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="mnist"):
	if source or not train:
		data = sio.loadmat(path+'/dslr.mat')
		if train: x, y = data["X"], data["y"]
		else: x, y = data["X_test"], data["y_test"]
		data.clear()
		return DataLoader( OFFICEDataset(x, y, transform), batch_size=batch_size, shuffle=True )

	labels_path = "{}{}_to_DSLR_{}shots_FADA.pkl".format(path, source_name.upper(), n)
	if not os.path.exists(labels_path):
		file = bzip2.BZ2File(labels_path, 'wb')
		data = sio.loadmat(path+'/dslr.mat')
		x_train, y_train = data["X"], data["y"]
		data.clear()
		data = OFFICEDataset(x_train, y_train, transform)

		labels = {}
		for i in range(31): labels[i] = []
		for i in range(len(data)):
			labels[ data[i][1].item() ].append(i)

		x, y = [], []
		for label, index in labels.items():
			random.shuffle(index)
			for i in range(n):
				j = index[i]
				x.append( data[j][0] )
				y.append( data[j][1] )

		pickle.dump( (x,y), file )
		file.close()
		return x, y

	file  = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	return x, y



# ------------------------------- WEBCAM dataset ------------------------------------- #


def get_webcam(path='./data/WEBCAM', train=True, transform=transforms.Compose([transforms.ToTensor()]), batch_size=256, source=True, n=1, source_name="mnist"):
	if source or not train:
		data = sio.loadmat(path+'/webcam.mat')
		if train: x, y = data["X"], data["y"]
		else: x, y = data["X_test"], data["y_test"]
		data.clear()
		return DataLoader( OFFICEDataset(x, y, transform), batch_size=batch_size, shuffle=True )

	labels_path = "{}/{}_to_WEBCAM_{}shots_FADA.pkl".format(path, source_name.upper(), n)
	if not os.path.exists(labels_path):
		file = bzip2.BZ2File(labels_path, 'wb')
		data = sio.loadmat(path+'/webcam.mat')
		x_train, y_train = data["X"], data["y"]
		data.clear()
		data = OFFICEDataset(x_train, y_train, transform)

		labels = {}
		for i in range(31): labels[i] = []
		for i in range(len(data)):
			labels[ data[i][1].item() ].append(i)

		x, y = [], []
		for label, index in labels.items():
			random.shuffle(index)
			for i in range(n):
				j = index[i]
				x.append( data[j][0] )
				y.append( data[j][1] )

		pickle.dump( (x,y), file )
		file.close()
		return x, y

	file  = bzip2.BZ2File(labels_path, 'rb')
	(x,y) = pickle.load(file, encoding='bytes')
	file.close()
	return x, y