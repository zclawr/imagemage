import numpy as np
from . import util
from . import convolutional

def puzzle(img_np, div):
    H, W, _ = img_np.shape
    block_size = int(np.floor(W / div))
    y_iter = int(np.ceil(H/block_size))
    result = np.copy(img_np)
    for i in range(y_iter):
        for j in range(div):
            flip = ((i % 2) + j) % 2 == 1
            axis_to_flip = i % 2
            if flip:
                low_h = j * block_size
                high_h = np.minimum((j+1) * block_size, W-1)
                low_w = i * block_size
                high_w = np.minimum((i+1) * block_size, H-1) 
                flipped_block = np.flip(img_np[low_w : high_w, low_h : high_h, :], axis=axis_to_flip)
                result[low_w : high_w, low_h : high_h, :] = flipped_block
    return result

def checker_invert(img_np, div):
    H, W, _ = img_np.shape
    block_size = int(np.floor(W / div))
    y_iter = int(np.ceil(H/block_size))
    for i in range(y_iter):
        for j in range(div):
            flip = ((i % 2) + j) % 2 == 1
            if flip:
                low_h = j * block_size
                high_h = np.minimum((j+1) * block_size, W-1)
                low_w = i * block_size
                high_w = np.minimum((i+1) * block_size, H-1) 
                flipped_block = util.invert(img_np[low_w : high_w, low_h : high_h, :])
                img_np[low_w : high_w, low_h : high_h, :] = flipped_block
    return img_np

def checker_hardlpf(img_np, div):
    H, W, _ = img_np.shape
    block_size = int(np.floor(W / div))
    y_iter = int(np.ceil(H/block_size))
    for i in range(y_iter):
        for j in range(div):
            flip = ((i % 2) + j) % 2 == 1
            if flip:
                low_h = j * block_size
                high_h = np.minimum((j+1) * block_size, W-1)
                low_w = i * block_size
                high_w = np.minimum((i+1) * block_size, H-1) 
                flipped_block = convolutional.ideal_lpf(img_np[low_w : high_w, low_h : high_h, :], 20)
                img_np[low_w : high_w, low_h : high_h, :] = flipped_block
    return img_np

def checker_chrotate(img_np, div):
    H, W, _ = img_np.shape
    block_size = int(np.floor(W / div))
    y_iter = int(np.ceil(H/block_size))
    for i in range(y_iter):
        for j in range(div):
            flip = ((i % 2) + j) % 2 == 1
            if flip:
                low_h = j * block_size
                high_h = np.minimum((j+1) * block_size, W-1)
                low_w = i * block_size
                high_w = np.minimum((i+1) * block_size, H-1) 
                flipped_block, _ = util.rotate_channels(img_np[low_w : high_w, low_h : high_h, :])
                img_np[low_w : high_w, low_h : high_h, :] = flipped_block
    return img_np