# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:24:47 2025

@author: cshen
"""

import torch
import torch.nn as nn

class AdaptiveTopoFusion(nn.Module):
    def __init__(self, c_dim: int, t_dim: int,
                 hidden: int,
                 mode: str = 'gate',
                 vector_gate: bool = True,
                 dropout: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()
        assert mode in ['gate', 'film', 'attn']
        self.mode = mode
        self.c_dim = c_dim
        self.t_dim = t_dim
        self.vector_gate = vector_gate

        self.t_norm = nn.LayerNorm(t_dim) if use_layernorm else nn.Identity()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.t_to_c = nn.Linear(hidden, c_dim)

        if mode == 'gate':
            if vector_gate:
                self.g_head = nn.Sequential(
                    nn.Linear(hidden + c_dim, c_dim),
                    nn.Sigmoid()
                )
            else:
                self.g_head = nn.Sequential(
                    nn.Linear(hidden + c_dim, 1),
                    nn.Sigmoid()
                )

        elif mode == 'film':
            self.gamma = nn.Linear(hidden, c_dim)
            self.beta  = nn.Linear(hidden, c_dim)

        elif mode == 'attn':
            self.q_proj = nn.Linear(c_dim, c_dim, bias=False)
            self.k_proj = nn.Linear(hidden, c_dim, bias=False)
            self.v_proj = nn.Linear(hidden, c_dim, bias=False)
            self.out = nn.Sequential(
                nn.Linear(c_dim, c_dim),
                nn.Dropout(dropout)
            )

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h: [B, C], t: [B, T]
        return: [B, C]
        """
        B, C = h.size(0), h.size(1)
        t_ = self.t_norm(t)                 
        zt = self.t_mlp(t_)                 # [B, hidden]
        tc = self.t_to_c(zt)               # [B, C] 

        if self.mode == 'gate':
            g_in = torch.cat([zt, h], dim=1)  # [B, hidden+C]
            g = self.g_head(g_in)             # [B, C] 或 [B, 1]
            if not self.vector_gate:
                g = g.expand(-1, C)           
            f = (1.0 - g) * h + g * tc
            return f

        elif self.mode == 'film':
            # FiLM: f = (1+γ)*h + β， 
            gamma = torch.tanh(self.gamma(zt))   # [-1,1] 
            beta  = self.beta(zt)
            f = (1.0 + gamma) * h + beta
            return f

        else:  # 'attn'
            Q = self.q_proj(h).unsqueeze(1)      # [B, 1, C]
            K = self.k_proj(zt).unsqueeze(1)     # [B, 1, C]
            V = self.v_proj(zt).unsqueeze(1)     # [B, 1, C]
            att = torch.softmax((Q @ K.transpose(1,2)) / (C ** 0.5), dim=-1)  # [B,1,1]
            ctx = (att @ V).squeeze(1)           # [B, C]
            f = h + self.out(ctx)                
            return f


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, nb_filters, kernel_sizes=(9,19,39), bottleneck=32):
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
    def __init__(self, in_ch, dim_topo, n_classes, nb_filters, depth, hidden,mode):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(InceptionBlock(in_ch if i==0 else nb_filters*4, nb_filters=nb_filters))
        self.features = nn.Sequential(*blocks)       
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        c_dim = nb_filters*4
        
        self.fuse = AdaptiveTopoFusion(
            c_dim = c_dim, t_dim= dim_topo, hidden=hidden, mode=mode,
            vector_gate=True, dropout=0.1,
            use_layernorm=True
        )
        
        
        self.head = nn.Sequential(
            nn.Linear(c_dim, 2*c_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(2*c_dim, n_classes)
        )
        
        
    def forward(self, x, topx):
        z = self.features(x)
        z = self.gap(z).squeeze(-1)
        f = self.fuse(z, topx)
        # f = z
        logits = self.head(f)
        
        return logits
