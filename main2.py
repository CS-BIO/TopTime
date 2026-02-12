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
import argparse

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from utils_ import topo_vector_from_pd
from utils_ import betti0_curve
from utils_ import sliding_topo_channel

from model import InceptionTime

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

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

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str, default='Fish', help='')
    parser.add_argument('--cat_mode', type=str, default='attn', help=['gate', 'film', 'attn'])
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--batch', type=int, default=64, help='Number of batch size')
    parser.add_argument('--hidden', type=int, default=128, help='')
    parser.add_argument('--conv_depth', type=int, default=7, help='')
    parser.add_argument('--nb_filters', type=int, default=32, help='')
    parser.add_argument('--classes', type=int, help='')
    parser.add_argument('--dim_input', type=int, help='')
    parser.add_argument('--dim_topo', type=int, help='')
    # parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    # parser.add_argument('--patience', type=int, default=10, help='Patience')
    
    args = parser.parse_args()
    
    
    ucr_dataset = args.dataset
    
    train_path = f"UCR_data/UCRArchive_2018/{ucr_dataset}/{ucr_dataset}_TRAIN.tsv"
    test_path = f"UCR_data/UCRArchive_2018/{ucr_dataset}/{ucr_dataset}_TEST.tsv"
    
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)
    
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    
    
    if os.path.exists(f"Topo_feature/{ucr_dataset}_topo_tr_t.npy") and os.path.exists(f"Topo_feature/{ucr_dataset}_topo_te_t.npy"):
        topo_tr_t = torch.from_numpy(np.load(f"Topo_feature/{ucr_dataset}_topo_tr_t.npy")).float()
        topo_te_t = torch.from_numpy(np.load(f"Topo_feature/{ucr_dataset}_topo_te_t.npy")).float()
    else:
        topo_tr = build_topo_vecs(X_train)
        topo_te = build_topo_vecs(X_test)

        topo_tr_t = torch.from_numpy(topo_tr).float()
        topo_te_t = torch.from_numpy(topo_te).float()
    
        np.save(f"Topo_feature/{ucr_dataset}_topo_tr_t.npy",topo_tr_t)
        np.save(f"Topo_feature/{ucr_dataset}_topo_te_t.npy",topo_te_t)
        
    
    args.dim_topo = topo_tr_t.shape[1]
    
    Xtr_t = torch.from_numpy(X_train).float().unsqueeze(1)
    Xte_t = torch.from_numpy(X_test).float().unsqueeze(1)
    
    all_y = np.concatenate([y_train, y_test])
    classes, y_all_enc = np.unique(all_y, return_inverse=True)
    
    
    ytr_enc = y_all_enc[:len(y_train)]
    yte_enc = y_all_enc[len(y_train):]
    
    ytr_t = torch.from_numpy(ytr_enc).long()
    yte_t = torch.from_numpy(yte_enc).long()
    
    num_classes = int(max(ytr_enc.max(), yte_enc.max()) + 1)
    args.classes = num_classes

    from sklearn.utils.class_weight import compute_class_weight
    cls_w = compute_class_weight("balanced", classes=np.arange(args.classes), y=ytr_enc)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32).to(device)
    
    all_results = []
    for aa in range(5):
        model = InceptionTime(in_ch=1,
                              dim_topo = args.dim_topo,
                              n_classes = args.classes, 
                              nb_filters = args.nb_filters, 
                              depth = args.conv_depth, 
                              hidden = args.hidden,
                              mode = args.cat_mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=cls_w_t)
        criterion_MSE = nn.MSELoss()
        
        # print(Xtr_t.shape)
        # print(topo_tr_t.shape)
        train_loader = DataLoader(TensorDataset(Xtr_t,topo_tr_t, ytr_t), batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(TensorDataset(Xte_t,topo_te_t, yte_t), batch_size=args.batch, shuffle=False)  
        
        best_acc, patience, wait = 0.0, 500, 0
        for epoch in range(args.epochs):
            model.train()
            for xb,tb,yb in train_loader:
                xb,tb,yb = xb.to(device),tb.to(device), yb.to(device)
                opt.zero_grad() 
                assert xb.ndim == 3 and xb.shape[1] == 1, f"bad shape {xb.shape}, need [B,1,L]"
                logits = model(xb,tb)
                
                pred_train = logits.argmax(1).cpu().numpy()
                acc_train = (pred_train == yb.cpu().numpy()).mean()
             
                loss = criterion(logits, yb)
                loss.backward() 
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                opt.step() 
        
          
            model.eval()
            with torch.no_grad():
                logits = model(Xte_t.to(device),topo_te_t.to(device))
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
