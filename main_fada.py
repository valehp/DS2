# -*- coding: utf-8 -*-

#from training import pretrain, train_discriminator, train
#from data import sample_groups
import torch
import pandas as pd
from FADA.datasets import *
from FADA.dataloaders import *
from FADA.train import *
from torchvision import transforms
from torch.utils.data import DataLoader


class Switcher:
    def __init__(self, path='./data/', office=False):
        self.p = path
        self.office = office
        if office:
            self.t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.CenterCrop(227),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.t = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.Resize( (16, 16) ),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize( (0.5,), (0.5,) ),
                ])


        self.options = {
            'mnist': lambda p, train, trans, bs, is_source, n, sname: get_mnist(p+'MNIST', train, trans, bs, is_source, n, sname),
            'usps' : lambda p, train, trans, bs, is_source, n, sname: get_usps(p+'USPS', train, trans, bs, is_source, n, sname),
            'svhn' : lambda p, train, trans, bs, is_source, n, sname: get_svhn(p+'SVHN', train, trans, bs, is_source, n, sname),
            'amazon': lambda p, train, trans, bs, is_source, n, sname: get_amazon(p+'AMAZON', train, trans, bs, is_source, n, sname),
            'webcam': lambda p, train, trans, bs, is_source, n, sname: get_webcam(p+'WEBCAM', train, trans, bs, is_source, n, sname),
            'dslr'  : lambda p, train, trans, bs, is_source, n, sname: get_dslr(p+'DSLR', train, trans, bs, is_source, n, sname),
        }

    def get_data(self, opt, train, bs, is_source, n, sname):
        return self.options[opt](self.p, train, self.t, bs, is_source, n, sname)


def train_FADA(n, reps, source, target, num_class, epochs, gamma=0.2, office=False):
    """
    - n: num of target samples
    - reps: num of repetitions of the training
    - source: name of the source domain
    - target: name of the target domain
    - epochs: list of each step epochs => [epochs_step1, epochs_step2, epochs_step3]
    """
    OPTS = Switcher(office=office)
    print("source: ", source, " - target: ", target )

    train_mode = True if source != 'svhn' else 'train'
    test_mode  = False if source != 'svhn' else 'test' 

    source_loader = OPTS.get_data(source, train_mode, 256, True, n, source)                                     # Get train dataloader from source domain for step1
    test_source_loader = OPTS.get_data(source, test_mode, 128, True, n, source)                                 # Get test dataloader from source domain for step1 
    XT, YT = OPTS.get_data(target, True if target != 'svhn' else 'train', 256, False, n, source)                # Get target labeled samples
    test_target_loader = OPTS.get_data(target, False if target != 'svhn' else 'test' , 256, True, n, source)    # Get train dataloader from target domain for step3


    groups, labels = sample_groups(n, source_loader, XT, YT, num_class) 
    print(len(groups[0]), len(source_loader), len(test_source_loader))

    X1, X2, Y1, Y2, T, F = [], [], [], [], [], []
    for j in range(4): # group j
        for i in range(len(groups[j])):
            x1, x2 = groups[j][i]
            y1, y2 = labels[j][i]

            x1, x2 = x1.type(torch.FloatTensor), x2.type(torch.FloatTensor)
            y1, y2 = y1, y2
            truth  = j
            fake   = 0 if(j==0 or j==1) else 2

            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)
            T.append(truth)
            F.append(fake)
    dcd_dataloader = DataLoader( DCD_dataset(X1, Y1, X2, Y2, T, F), batch_size=256, shuffle=True )

    accs = []
    ba   = []
    cuda = torch.cuda.is_available()
    learning_rate = 1e-4
    with open('{}_to_{}.txt'.format(source.upper(), target.upper()), 'w') as f:
        f.write( " ========== {} to {} , {}-shots ========== ".format(source.upper(), target.upper(), n) )

    for rep in range(reps):
        print( '----- {} to {} ----- rep {} for {}-shots ----- \n'.format(source.upper(), target.upper(), rep, n) )
        
        print( " ========== STEP 1 Traing g and h ========== " )
        if office: gh = pretrain_office(source_loader, test_source_loader, epochs[0], 256, cuda, source, LR=learning_rate)
        else: gh = pretrain(source_loader, test_source_loader, epochs[0], cuda, source, office)

        print( "\n ========== STEP 2 Traing DCD ========== " )
        discriminator = train_discriminator(g_h=gh, train_loader=dcd_dataloader, task='{}to{}'.format(source.upper(), target.upper()), n_target_samples=n, cuda=cuda, epochs=epochs[1], office=office)
        
        print( "\n ========== STEP 2 Training loop -> Train g & h while DCD frozen / Train DCD while g & h frozen  ========== " )
        acc, best_acc = train(gh, discriminator, dcd_dataloader, test_target_loader, n, cuda, epochs[2], gamma, tol=epochs[3], file_name='{}_to_{}.txt'.format(source.upper(), target.upper()), LR=learning_rate) # 0.2 -> value for gamma
        
        del gh
        del discriminator
        accs.append(acc)
        ba.append(best_acc)
        print( '\t Accuracy : ', accs[-1], ba[-1] )
    
    print( '\t Mean accuracy : ', np.mean(accs), ' - std: ', np.std(accs) )
    with open('resultados_FADA_'+source+'.txt', 'a') as file:
        file.write( 'Task: '+source+' to '+target+' for '+str(n)+'-shots:\n' )
        file.write( '\t Accuracys: {} -- Best acc: {}\n'.format(accs, ba) )
        file.write( '\t Mean accuracy: {}\n'.format( np.mean(accs) ) )
        file.write( '\t STD: {}\n\n'.format( np.std(accs) ) )


shots   = list(range(1, 8))
reps    = 1
epochs  = [500, 1000, 10000, 500] # epoch step1, step2, step3, tol for early stopping
classes = 31
office  = True
gamma   = 0.01


source = 'dslr'
target = 'amazon'
task   = '{} to {}'.format(source.upper(), target.upper())
n      = 3
train_FADA(n=n, reps=reps, source=source, target=target, num_class=classes, epochs=epochs, gamma=gamma, office=office)