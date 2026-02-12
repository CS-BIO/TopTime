# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 16:45:01 2025

@author: cshen
"""

import numpy as np
from ripser import ripser
from persim import PersImage

def takens_embed(x, m=3, tau=None):
    L = len(x)
    if tau is None:
        tau = max(1, L // 32)
    M = L - (m-1)*tau
    X = np.stack([x[i:i+M] for i in range(0, m*tau, tau)], axis=1)  # [M, m]
    return X

def safe_normalize_life(life, eps=1e-12):
    life = np.asarray(life, dtype=float)
    life = np.where(np.isfinite(life), life, 0.0)
    life = np.clip(life, 0.0, None)

    s = life.sum()
    if not np.isfinite(s) or s <= eps:
        return np.zeros_like(life)
    return life / s


def _clean_diag(H):
    H = np.asarray(H, dtype=float)
    if H.size == 0:
        return H
    finite = np.isfinite(H).all(axis=1)
    H = H[finite]
    if H.size == 0:
        return H
    b = np.maximum(H[:, 0], 0.0)
    d = np.maximum(H[:, 1], 0.0)
    keep = d > b
    if not np.any(keep):
        return np.zeros((0, 2), dtype=float)
    return np.stack([b[keep], d[keep]], axis=1)

def _safe_pers_image(H, pixels=(16, 16), spread=0.1):
    H = _clean_diag(H)
    if H.size == 0:
        return np.zeros(pixels, dtype=float)
    y = H[:, 1] - H[:, 0]
    ymax = np.nanmax(y) if y.size else 0.0
    if not np.isfinite(ymax) or ymax <= 0:
        return np.zeros(pixels, dtype=float)
    pim = PersImage(pixels=list(pixels), spread=spread, kernel_type="gaussian", weighting_type="linear")
    return pim.transform(H)


def topo_vector_from_pd(x, m=3, tau=None, n_bins=16):
    """
    计算拓扑向量：H0/H1的统计量 + H0的Persistence Image（扁平化）
    """
    X = takens_embed(x, m=m, tau=tau)   # [M, m]
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        vec8 = np.zeros(8, dtype=np.float32)
        pi0 = np.zeros(n_bins * n_bins, dtype=np.float32)
        return np.concatenate([vec8, pi0], axis=0)

    row_finite = np.isfinite(X).all(axis=1)
    X = X[row_finite]
    if X.shape[0] < 2:
        vec8 = np.zeros(8, dtype=np.float32)
        pi0 = np.zeros(n_bins * n_bins, dtype=np.float32)
        return np.concatenate([vec8, pi0], axis=0)

    dgms = ripser(X, maxdim=1)['dgms']
    H0 = dgms[0] if len(dgms) >= 1 else np.zeros((0, 2), dtype=float)
    H1 = dgms[1] if len(dgms) >= 2 else np.zeros((0, 2), dtype=float)

    H0 = _clean_diag(H0)
    H1 = _clean_diag(H1)

    def stats(H):
        if H.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        life = (H[:, 1] - H[:, 0]).clip(min=0.0)
        if not np.any(np.isfinite(life)):
            return [0.0, 0.0, 0.0, 0.0]
        life = np.where(np.isfinite(life), life, 0.0)
        s = float(np.sum(life))
        m3 = float(np.sort(life)[-3:].sum()) if life.size >= 3 else s
        mx = float(np.max(life)) if life.size else 0.0
        if s <= 1e-12:
            ent = 0.0
        else:
            p = safe_normalize_life(life)
            ent = float((-(p) * np.log(p + 1e-12)).sum())
        return [s, mx, m3, ent]

    feat = stats(H0) + stats(H1)  

    PI0 = _safe_pers_image(H0, pixels=(n_bins, n_bins), spread=0.1)
    pi0_flat = PI0.ravel().astype(np.float32)

    vec = np.array(feat, dtype=np.float32)
    return np.concatenate([vec, pi0_flat], axis=0)  # (8 + n_bins^2,)


def betti0_curve(x, n_thresh=64):
    """子水平集阈值扫描得到 Betti_0(θ) 曲线（全局向量，晚融合用）"""
    vmin, vmax = float(x.min()), float(x.max())
    thetas = np.linspace(vmin, vmax, n_thresh)
    curv = []
    for th in thetas:
        mask = (x <= th).astype(np.int32)
        comps = int((np.diff(np.pad(mask, (1,1))) == 1).sum())
        curv.append(float(comps))
    curv = np.array(curv, dtype=np.float32)
    return (curv - curv.mean()) / (curv.std() + 1e-8)

def sliding_topo_channel(x, win=64, step=4):
    L = len(x)
    out = np.zeros(L, dtype=np.float32)
    if win > L:  
        win = L
    for s in range(0, L - win + 1, step):
        seg = x[s:s+win].reshape(-1,1)    
        H0 = ripser(seg, maxdim=0)['dgms'][0]
        life = (H0[:,1] - H0[:,0]).clip(min=0)
        val = float(life.sum()) if life.size else 0.0
        out[s:s+win] += val
    if out.max() > 0:
        out = out / out.max()
    return out







