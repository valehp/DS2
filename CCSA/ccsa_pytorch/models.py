import torch
from torch import nn

class G(nn.Module):
	# Encoder
	def __init__(self):
		super(G, self).__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(1, 6, 5),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(6, 16, 5),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.linear= nn.Sequential(
			nn.Flatten(),
			nn.Linear(16, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
		)

	def forward(self, x):
		out = self.convs(x)
		out = self.linear(out)
		return out


class H(nn.Module):
	# Classifier
	def __init__(self, D_in=84, D_out=10):
		super(H, self).__init__()
		self.layer = nn.Sequential(
			nn.Linear(D_in, D_out),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.layer(x)



class ModelPaper(nn.Module):
	def __init__(self):
		super(ModelPaper, self).__init__()
		self.encoder = G()
		self.classifier = H()

	def forward_encoder(self, x):
		return self.encoder(x)

	def forward_once(self, x):
		out_e = self.encoder(x)
		out_c = self.classifier(out_e)
		return (out_e, out_c)

	def forward(self, x1, x2):
		out1 = self.forward_once(x1)
		out2 = self.forward_once(x2)
		return out1, out2



class ModelGihub(nn.Module):
	def __init__(self):
		super(ModelPaper, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, 3),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Dropout2d(0.25)
			nn.Flatten(),
			nn.Linear(16, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
		)
		self.classifier = H()

	def forward_encoder(self, x):
		return self.encoder(x)

	def forward_once(self, x):
		out_e = self.encoder(x)
		out_c = self.classifier(out_e)
		return (out_e, out_c)

	def forward(self, x1, x2):
		out1 = self.forward_once(x1)
		out2 = self.forward_once(x2)
		return out1, out2

# =============== Loss functions =============== #

def SemanticAlignmentLoss(z1, z2):
	"""
	- z1 = g(X^s_a)
	- z2 = g(X^t_a)
	"""
	loss = torch.sum( torch.norm(z1-z2) )
	return 1/2 * loss

def SeparationLoss(z1, z2. m=1e-08):
	"""
	- z1 = g(X^s_a)
	- z2 = g(X^t_b)
	"""
	loss = torch.sum( torch.max( 0, m-torch.norm(z1-z2) ) )
	return 1/2 * loss

def CCSA(CL, SA, S, alpha=0.25):
	"""
	- CL: Classification Loss
	- SA: Semantic Alignment Loss
	- S : Separation Loss
	"""
	loss = alpha*CL + (1-alpha) * (SA + S)
	return loss

def CS(CL, SL):
	"""
	- CL: Classification Loss
	- S : Separation Loss
	"""
	return CL + S