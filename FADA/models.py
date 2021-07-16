# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

# --------------------- NN for digits domains --------------------- #

'''
    Domain-Class Discriminator (see (3) in the paper)
    Takes in the concatenated latent representation of two samples from
    G1, G2, G3 or G4, and outputs a class label, one of [0, 1, 2, 3]
'''
class DCD(nn.Module):
    def __init__(self, H=64, D_in=256):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.softmax(self.out(out), dim=1)

''' Called h in the paper. Gives class predictions based on the latent representation '''
class Classifier(nn.Module):
    def __init__(self, D_in=64, D_out=10):
        super(Classifier, self).__init__()
        self.out = nn.Linear(D_in, D_out)

    def forward(self, x):
        return F.softmax(self.out(x), dim=1)

''' 
    Creates latent representation based on data. Called g in the paper.
    Like in the paper, we use g_s = g_t = g, that is, we share weights between target
    and source representations.
    Model is as specified in section 4.1. See https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 64)
        #self.flat  = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        #out = self.flat(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def forward_encoder(self, x):
        out = self.encoder(x)
        return out

    def forward_once(self, x):
        out_e = self.encoder(x)
        out_c = self.classifier(out_e)
        return (out_e, out_c)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2



# --------------------- NN for Office domains --------------------- #

class G2(nn.Module):
    def __init__(self, pretrained=True):
        super(G2, self).__init__()
        model_alexnet = torchvision.models.alexnet(pretrained=pretrained)
        #print(model_alexnet)
        #model_alexnet.classifier[6].out_features = classes
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.FC(x)
        #x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return x


class H2(nn.Module):
    def __init__(self, in_dim=128, n_class=31):
        super(H2, self).__init__()    
        self.fc = nn.Linear(in_dim, n_class)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc(x)
        out = self.sm(out)
        return out


class NET(nn.Module):
    def __init__(self, in_dim=128, n_class=31):
        super(NET, self).__init__()    
        self.feature_extractor = G2()
        self.classifier        = H2()

    def forward_encoder(self, x):
        return self.feature_extractor(x)

    def forward_once(self, x):
        z = self.forward_encoder(x)
        y = self.classifier(z)
        return (z, y)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class DCD_office(nn.Module):
    def __init__(self, H1=128, H2=64, D_in=256):
        super(DCD_office, self).__init__()
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.out = nn.Linear(H2, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out) 
        return F.softmax(self.out(out), dim=1)

# ---- G with AlexNet encoder ---- #
class SiameseNet_office(nn.Module):
    def __init__(self, pret=True, n_class=31):
        super(SiameseNet_office, self).__init__()    
        alexnet = torchvision.models.alexnet(pretrained=pret)
        for param in alexnet.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential( *list(alexnet.features._modules.values())[:] )
        self.FC = nn.Sequential(
                nn.Flatten(),
                nn.Linear(9216, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
            )
        self.classifier = Classifier(D_in=128, D_out=n_class)

    def forward_encoder(self, x):
        out = self.encoder(x)
        out = self.FC(out)
        return out

    def forward_once(self, x):
        out_e = self.forward_encoder(x)
        out_c = self.classifier( out_e )
        return (out_e, out_c)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)