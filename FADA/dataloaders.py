# -*- coding: utf-8 -*-
import torchvision
from torchvision import transforms
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import pickle, gzip

''' 
    Samples uniformly groups G1 and G3 from D_s x D_s and groups G2 and G4 from D_s x D_t  
'''
def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)
    n=X_t.shape[0] 
    #shuffle order
    classes = torch.unique(Y_t)
    shuf_classes=classes[torch.randperm(len(classes))]
    
    class_num=shuf_classes.shape[0]
    shot=n//class_num
    #shot = n
    
    def get_idx(map_labels, labels):
        for i in range(len(labels)): map_labels[labels[i].item()].append(i)
        return map_labels
    def g1_sample(Y, ids, X): # X^s_i, X^s_i
        i1 = random.choice(ids)
        i2 = random.choice(ids)
        return (X[i1], X[i2]), (Y[i1], Y[i2])
    def g2_sample(Ys, Yt, idx_s, idx_t, Xs, Xt): # X^s_i, X^t_i
        s = random.choice(idx_s)
        t = random.choice(idx_t)
        return ( Xs[s], Xt[t] ), (Ys[s], Yt[t])
    def g3_sample(X, Y, idx_i, idx_j): # X^s_i , X^s_j
        i = random.choice(idx_i)
        j = random.choice(idx_j)
        return (X[i], X[j]), (Y[i], Y[j])
    def g4_sample(Xs, Xt, Ys, Yt, idx_s, idx_t): # X^s_i, X^t_j
        i = random.choice(idx_s)
        j = random.choice(idx_t)
        return (Xs[i], Xt[j]), (Ys[i], Yt[j])
    
    s_idx = torch.arange(0, len(X_s))
    t_idx = torch.arange(0, len(X_t))
    s_labels = np.unique(Y_s)
    t_labels = np.unique(Y_t)
    s_map = dict(zip(s_labels, [ [] for i in range(len(s_labels)) ]))
    t_map = dict(zip(t_labels, [ [] for i in range(len(t_labels)) ]))
    
    s_map = get_idx(s_map, Y_s)
    t_map = get_idx(t_map, Y_t)
    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []
    for i in s_labels:
        for j in t_labels:
            tmp1 = g1_sample(Y_s, s_map[i], X_s)
            tmp2 = g2_sample(Y_s, Y_t, s_map[i], t_map[j], X_s, X_t)
            ii = i
            while ii == j or ii not in s_labels: ii = random.choice(s_labels)
            tmp3 = g3_sample(X_s, Y_s, s_map[ii], s_map[j])
            ii = i
            while ii == j or ii not in s_labels: ii = random.choice(s_labels)
            tmp4 = g4_sample(X_s, X_t, Y_s, Y_t, s_map[ii], t_map[j])
            G1.append(tmp1[0])
            G2.append(tmp2[0])
            G3.append(tmp3[0])
            G4.append(tmp4[0])
            Y1.append(tmp1[1])
            Y2.append(tmp2[1])
            Y3.append(tmp3[1])
            Y4.append(tmp4[1])
    groups = [G1, G2, G3, G4]
    labels = [Y1, Y2, Y3, Y4]
    # Make sure we sampled enough samples
    for g in groups:
        assert(len(g) > n)
    return groups, labels

# ---- Call the sample groups for G1, G2, G3, G4 ---- #
def sample_groups(n_target_samples, source_loader, X_t, y_t, num_classes):
    X_s, y_s = [], []
    for imgs, labels in source_loader:
        for i in range(len(labels)):
            X_s.append(imgs[i])
            y_s.append(labels[i])
    X_t = torch.stack(X_t)
    y_t = torch.stack(y_t)
    return create_groups(X_s, y_s, X_t, y_t)
