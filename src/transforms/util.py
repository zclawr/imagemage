import numpy as np

def mono(img_np):
    # Uses Luminance algorithm
    R_factor = 0.299
    G_factor = 0.587
    B_factor = 0.114
    mono = (R_factor * img_np[:,:,0]) + (G_factor * img_np[:,:,1]) + (B_factor * img_np[:,:,2])
    return np.repeat(mono[:, :, np.newaxis], 3, -1)

def rotate_channels(img_np):
    brg = np.zeros_like(img_np)
    brg[:,:,0] = img_np[:,:,2]
    brg[:,:,1] = img_np[:,:,0]
    brg[:,:,2] = img_np[:,:,1]

    gbr = np.zeros_like(img_np)
    gbr[:,:,0] = img_np[:,:,1]
    gbr[:,:,1] = img_np[:,:,2]
    gbr[:,:,2] = img_np[:,:,0]
    return brg, gbr

def invert(img_np, pass_thru=0):
    fill = get_max_value(img_np)
    max = np.full(shape=img_np.shape, fill_value=fill)
    inverted = max - img_np
    if pass_thru == 0:
        return inverted
    else:
        res = np.array([inverted, img_np])
        return res

def modulo(img_np, mod, channel_linked=False):
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if channel_linked:
                if(img_np[i,j,0] >= mod and img_np[i,j,1] >= mod and img_np[i,j,2] >= mod):
                    img_np[i,j,:] = img_np[i,j,:] % mod
            else:
                for k in range(img_np.shape[2]):
                    img_np[i,j,k] = img_np[i,j,k] % mod
    return img_np

def subtract(imgs):
    min = get_min_value(imgs[0])
    res = imgs[0]
    for img in imgs:
        res -= img
    res = np.maximum(res, np.full_like(res, fill_value=min))
    return res

def add(imgs, cycle=0):
    max = get_max_value(imgs[0])
    res = np.zeros_like(imgs[0])
    for img in imgs:
        res += img
    if cycle==1:
        res = modulo(res, max)
    else:
        res = np.minimum(res, np.full_like(res, fill_value=max))
    return res

def divide():
    pass

def yflip(img):
    return np.flip(img, axis=0)

def xflip(img):
    return np.flip(img, axis=1)

def multiply(imgs):
    max = get_max_value(imgs[0])
    res = imgs[0]
    for i in range(1, imgs.shape[0]):
        res *= imgs[i]
    res = np.maximum(res, np.full_like(res, fill_value=max))
    return res

def bitwise_and(imgs):
    res = np.array(imgs[0], dtype=np.uint8)
    for i in range(1, imgs.shape[0]):
        res = res & np.array(imgs[i], dtype=np.uint8)
    return res

def bitwise_or(imgs):
    res = np.array(imgs[0], dtype=np.uint8)
    for i in range(1, imgs.shape[0]):
        res = res | np.array(imgs[i], dtype=np.uint8)
    return res

def bitwise_xor(imgs):
    res = np.array(imgs[0], dtype=np.uint8)
    for i in range(1, imgs.shape[0]):
        res = res ^ np.array(imgs[i], dtype=np.uint8)
    return res

def bitshift_left(img, shift):
    return np.array(img, dtype=np.uint8) << shift

def bitshift_right(img, shift):
    return np.array(img, dtype=np.uint8) >> shift

def get_max_value(img_np):
    if(img_np.dtype == np.uint8):
        fill = np.uint8(255)
    else:
        fill=np.finfo(img_np.dtype).max
    return fill

def get_min_value(img_np):
    if(img_np.dtype == np.uint8):
        fill = np.uint8(0)
    else:
        fill=np.finfo(img_np.dtype).min
    return fill