import torch
from torch.utils.data import DataLoader
import torchvision

def CreatePairs(source, target, batch_size=128):
	"""
	1. Create pairs for SA loss => same label, diff domains
	2. Create pairs for S  loss => diff label, diff domains
	"""
	def get_labels(data):
		l = set()
		for _, y in data:
			
	G1, G2 = [], []
	
	for xs, ys in source:
		for xt, yt in target:
			tmp = ( (xs, ys), (xt, yt) )
			if ys == yt: G1.append( tmp )
			else: G2.append( tmp )   

	G1 = DataLoader( G1, shuffle=True, batch_size=batch_size )
	G2 = DataLoader( G2, shuffle=True, batch_size=batch_size )
	return G1, G2
	

def PreTrain():
	pass

def trainCCSA():
	pass

def trainCS():
	pass