import numpy as np
from fbm import FBM

np.random.seed(42)

def sample_by_p(imgs, p):
    n,W,H,_ = imgs.shape
    res = np.zeros_like(imgs[0], dtype=imgs[0].dtype)
    for i in range(W):
        for j in range(H):
            k = int(np.floor(n * p[i,j]))
            res[i,j,:] = np.array(imgs[k,i,j,:], dtype=imgs[0].dtype)
    return res

def uniform(imgs):
    _,W,H,_ = imgs.shape
    p = np.random.uniform(size=(W,H))
    return sample_by_p(imgs, p)

def fracbm(imgs, hurst, vertical=0):
    hurst = hurst / 100
    epsilon = 0.0001
    _,W,H,_ = imgs.shape
    f = np.abs(np.array(FBM(n=W*H*2, hurst=hurst, length=1, method='daviesharte').fbm())[W*H:-1])
    if vertical == 0:
        f = np.reshape(f, shape=(W,H))
    else:
        f = np.transpose(np.reshape(f, shape=(H,W)))
    # Normalize path b/w [0,1]
    f = (f - np.min(f)) / (np.max(f) + epsilon)
    return sample_by_p(imgs, f)