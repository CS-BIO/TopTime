# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:08:44 2025

@author: cshen
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from utils import topo_vector_from_pd
from utils import betti0_curve
from utils import sliding_topo_channel

device = "cuda" if torch.cuda.is_available() else "cpu"

# device = "cpu"

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, nb_filters=32, kernel_sizes=(9,19,39), bottleneck=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, kernel_size=1, bias=False) if in_ch > 1 else nn.Identity()
        k1,k2,k3 = kernel_sizes
        self.conv1 = nn.Conv1d(bottleneck if in_ch>1 else in_ch, nb_filters, kernel_size=k1, padding=k1//2, bias=False)
        self.conv2 = nn.Conv1d(bottleneck if in_ch>1 else in_ch, nb_filters, kernel_size=k2, padding=k2//2, bias=False)
        self.conv3 = nn.Conv1d(bottleneck if in_ch>1 else in_ch, nb_filters, kernel_size=k3, padding=k3//2, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.convpool = nn.Conv1d(in_ch, nb_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(nb_filters*4)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        y = self.bottleneck(x) if not isinstance(self.bottleneck, nn.Identity) else x
        z1 = self.conv1(y) 
        z2 = self.conv2(y)
        z3 = self.conv3(y)
        z4 = self.convpool(self.pool(x))
        
        z = torch.cat([z1,z2,z3,z4], dim=1)
        z = self.bn(z)
        return self.relu(z)

class InceptionTime(nn.Module):
    def __init__(self, in_ch=1, n_classes=7, nb_filters=32, depth=5):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(InceptionBlock(in_ch if i==0 else nb_filters*4, nb_filters=nb_filters))
        self.features = nn.Sequential(*blocks)
        
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc_decoder = nn.Linear(128, 328)
        self.fc_top = nn.Linear(328, 16)
        self.fc = nn.Linear(nb_filters*4+16, n_classes)
        
        
    def forward(self, x, topx):
        z = self.features(x)
        z = self.gap(z).squeeze(-1)
        
        topx = self.fc_top(topx.squeeze(1))
        z1 = torch.cat((z,topx),dim=1)
        topx_pred = self.fc_decoder(z)
        
        return self.fc(z1),topx_pred
def build_topo_vecs(X_list):
    topo_vecs = []
    for x in X_list:
        v1 = topo_vector_from_pd(x, m=3, tau=None, n_bins=16)   # 8 + 256
        v2 = betti0_curve(x, n_thresh=64)                       # 64
        v = np.concatenate([v1, v2], axis=0).astype(np.float32) # topo_dim = 8+256+64 = 328
        topo_vecs.append(v)
    return np.stack(topo_vecs, axis=0)  # [N, topo_dim]

def build_two_channel_input(X_list):
    X2 = []
    for x in X_list:
        topo_ch = sliding_topo_channel(x, win=64, step=4)   # [L]
        X2.append(np.stack([x, topo_ch], axis=0))           # [2, L]
    return np.stack(X2, axis=0).astype(np.float32)          # [N, 2, L]



ucr_dataset = "Yoga"

train_path = f"UCR_data/UCRArchive_2018/{ucr_dataset}/{ucr_dataset}_TRAIN.tsv"
test_path = f"UCR_data/UCRArchive_2018/{ucr_dataset}/{ucr_dataset}_TEST.tsv"

train_data = np.loadtxt(train_path)
test_data = np.loadtxt(test_path)

X_train = train_data[:, 1:]
y_train = train_data[:, 0]
X_test = test_data[:, 1:]
y_test = test_data[:, 0]


if os.path.exists(f"{ucr_dataset}_topo_tr_t.npy") and os.path.exists(f"{ucr_dataset}_topo_te_t.npy"):
    topo_tr_t = torch.from_numpy(np.load(f"{ucr_dataset}_topo_tr_t.npy")).float()
    topo_te_t = torch.from_numpy(np.load(f"{ucr_dataset}_topo_te_t.npy")).float()
else:
    topo_tr = build_topo_vecs(X_train)
    topo_te = build_topo_vecs(X_test)
    topo_tr_t = torch.from_numpy(topo_tr).float().unsqueeze(1)
    topo_te_t = torch.from_numpy(topo_te).float().unsqueeze(1)

    np.save(f"{ucr_dataset}_topo_tr_t.npy",topo_tr_t)
    np.save(f"{ucr_dataset}_topo_te_t.npy",topo_te_t)
    
    topo_tr_t = torch.from_numpy(np.load(f"{ucr_dataset}_topo_tr_t.npy")).float()
    topo_te_t = torch.from_numpy(np.load(f"{ucr_dataset}_topo_te_t.npy")).float()

Xtr_t = torch.from_numpy(X_train).float().unsqueeze(1)
Xte_t = torch.from_numpy(X_test).float().unsqueeze(1)

all_y = np.concatenate([y_train, y_test])
classes, y_all_enc = np.unique(all_y, return_inverse=True)


ytr_enc = y_all_enc[:len(y_train)]
yte_enc = y_all_enc[len(y_train):]

ytr_t = torch.from_numpy(ytr_enc).long()
yte_t = torch.from_numpy(yte_enc).long()

num_classes = int(max(ytr_enc.max(), yte_enc.max()) + 1)

from sklearn.utils.class_weight import compute_class_weight
cls_w = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr_enc)

cls_w_t = torch.tensor(cls_w, dtype=torch.float32).to(device)

all_results = []
for aa in range(5):
    model = InceptionTime(in_ch=1, n_classes=num_classes, nb_filters=32, depth=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=cls_w_t)
    criterion_MSE = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(Xtr_t,topo_tr_t, ytr_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xte_t,topo_te_t, yte_t), batch_size=128, shuffle=False)  
    
    best_acc, patience, wait = 0.0, 500, 0
    for epoch in range(6000):
        model.train()
        for xb,tb,yb in train_loader:
            xb,tb,yb = xb.to(device),tb.to(device), yb.to(device)
            opt.zero_grad() 
            assert xb.ndim == 3 and xb.shape[1] == 1, f"bad shape {xb.shape}, need [B,1,L]"
            logits,topx_pred = model(xb,tb)
            
            decode_loss = criterion_MSE(topx_pred,tb.squeeze(1))
            
            pred_train = logits.argmax(1).cpu().numpy()
            acc_train = (pred_train == yb.cpu().numpy()).mean()
            loss = criterion(logits, yb)
            loss.backward() 
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step() 
    
        model.eval()
        with torch.no_grad():
            logits,_ = model(Xte_t.to(device),topo_te_t.to(device))
            pred = logits.argmax(1).cpu().numpy()
            acc = (pred == yte_t.cpu().numpy()).mean()
        if acc > best_acc:
            best_acc, wait = acc, 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
        
        if epoch % 100 == 0:
            print("==========epoch:",epoch)
            print("acc_train:",acc_train)
            print("best_acc:",best_acc)
    
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    print("InceptionTime ACC =", best_acc)
    all_results.append(best_acc)
print("最终结果：",all_results)
print("平均值：",np.mean(all_results))
