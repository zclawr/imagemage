import numpy as np
from PIL import Image
import os
import shutil
import argparse
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import binascii
import matplotlib.pyplot as plt

# Define constant pathing
root = os.path.dirname(__file__)
img_lib = os.path.join(root, 'img')

def add_img_to_library(img_path, img_name):
    img_file = img_path.split('/')[-1].strip()
    if img_name == '':
        img_name = img_file.split('.')[0]
    img_dir = os.path.join(img_lib, img_name)
    try:
        os.makedirs(img_dir, exist_ok=True)
        shutil.copy2(img_path, os.path.join(img_dir, 'source.png'))
    except Exception as e:
        print(e)

    return img_name

def save_transformed_img(transform_name, img_name, img_np):
    transform_path = os.path.join(os.path.dirname(get_img_path(img_name)), transform_name + '.png')
    Image.fromarray(img_np).save(transform_path)

def get_img_path(img_name):
    return os.path.join(img_lib, img_name, 'source.png')

def img_to_numpy(img_name):
    img_path = get_img_path(img_name)
    image = Image.open(img_path)
    return np.array(image, dtype=np.uint8)

def ecb_img(img_np):
    # Generate a random 16-byte key (for AES-128)
    key = get_random_bytes(16)
    if img_np.shape[0] % 2 == 1:
        img_np = img_np[:-1,:,:]
    if img_np.shape[1] % 2 == 1:
        img_np = img_np[:,:-1,:]
    print(img_np.shape)
    data = img_np.tobytes()
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(data)
    # Repack to same datatype and shape
    cipher_img = np.frombuffer(ciphertext, dtype=np.uint8)
    cipher_img = cipher_img.reshape(img_np.shape)

    decipher = AES.new(key, AES.MODE_ECB)
    deciphertext = decipher.decrypt(ciphertext)

    decipher_img = np.frombuffer(deciphertext, dtype=np.uint8)
    decipher_img = decipher_img.reshape(img_np.shape)
    return cipher_img, decipher_img

def low_rank_approximation_svd_img(img_np, k):
    """
    Computes the best rank-k approximation of a matrix A using SVD.
    """
    # Perform SVD over each RGB channel separately
    channels = img_np.shape[-1]
    svd_img = np.zeros(shape=img_np.shape, dtype= np.uint8)
    print(f'Beginning SVD with k={k}')
    for i in range(channels):
        A = img_np[:,:,i]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        print(f'Completed SVD on channel {i}')

        # Truncate to the first k singular values and vectors
        U_k = U[:, :k]
        s_k = np.diag(s[:k])
        Vt_k = Vt[:k, :]

        # Reconstruct the rank-k approximation
        A_k = U_k @ s_k @ Vt_k
        svd_img[:,:,i] = np.astype(A_k, np.uint8)
    return svd_img

def low_rank_approximation_svd_img_per_channel(img_np, k_r, k_g, k_b):
    """
    Computes the best rank-k approximation of a matrix A using SVD.
    """
    # Perform SVD over each RGB channel separately
    channels = img_np.shape[-1]
    svd_img = np.zeros(shape=img_np.shape, dtype= np.uint8)
    print(f'Beginning SVD with k_r={k_r}, k_g={k_g}, k_b={k_b}')
    ks = [k_r, k_g, k_b]
    for i in range(channels):
        A = img_np[:,:,i]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        print(f'Completed SVD on channel {i}')

        # Truncate to the first k singular values and vectors
        U_k = U[:, :ks[i]]
        s_k = np.diag(s[:ks[i]])
        Vt_k = Vt[:ks[i], :]

        # Reconstruct the rank-k approximation
        A_k = U_k @ s_k @ Vt_k
        svd_img[:,:,i] = np.astype(A_k, np.uint8)
    return svd_img

def to_mono(img_np):
    # Uses Luminance algorithm
    R_factor = 0.299
    G_factor = 0.587
    B_factor = 0.114
    mono = (R_factor * img_np[:,:,0]) + (G_factor * img_np[:,:,1]) + (B_factor * img_np[:,:,2])
    return np.repeat(mono[:, :, np.newaxis], 3, -1).astype(np.uint8)

def brownian_motion(n_steps, n_paths):
    dt = 1./10000
    # Generate random steps
    steps = np.sqrt(dt) * np.random.normal(0, 1, size=(n_steps, n_paths))
    # Cumulative sum to get paths
    path = np.cumsum(steps, axis=0)
    return path

def random_channel_walk(img_np):
    walks = brownian_motion(img_np.shape[1], img_np.shape[0] * 3)
    walks = np.abs(walks)
    walks = walks.reshape(img_np.shape)
    new_img = img_np * walks
    new_img = np.maximum(np.minimum(new_img, np.full_like(walks, 255)), np.full_like(walks, 0))
    new_img = np.astype(new_img, np.uint8)
    return new_img #np.astype(new_img, np.uint8)

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

def invert(img_np):
    max = np.full(shape=img_np.shape, fill_value=255, dtype=np.uint8)
    inverted = max - img_np
    return inverted

def process_img(img_name, args):
    img = img_to_numpy(img_name)
    if args.ecb:
        ecb, d_ecb = ecb_img(img)
        save_transformed_img('ecb', img_name, ecb)
    if args.svd:
        for k_r in range(8):
            for k_g in range(8):
                for k_b in range(8):
                    svd_k = low_rank_approximation_svd_img_per_channel(img, 2 ** k_r, 2 ** k_g, 2 ** k_b)
                    print(svd_k.shape)
                    save_transformed_img(f'svd_r{2 ** k_r}_g{2 ** k_g}_b{2 ** k_b}', img_name, svd_k)
    if args.mono:
        mono = to_mono(img)
        save_transformed_img('mono', img_name, mono)
    if args.rcwalk:
        rcwalk = random_channel_walk(img)
        save_transformed_img('rcwalk', img_name, rcwalk)
    if args.chrotate:
        brg, gbr = rotate_channels(img)
        save_transformed_img('chrotate_brg', img_name, brg)
        save_transformed_img('chrotate_gbr', img_name, gbr)
    if args.invert:
        inv = invert(img)
        save_transformed_img('inverted', img_name, inv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--add', default='', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--input', default='', type=str)
    parser.add_argument('--ecb', action="store_true")
    parser.add_argument('--svd', action="store_true")
    parser.add_argument('--mono', action="store_true")
    parser.add_argument('--rcwalk', action="store_true")
    parser.add_argument('--chrotate', action="store_true")
    parser.add_argument('--invert', action="store_true")
    args = parser.parse_args()

    img_name = args.input
    if len(args.add) > 0:
        img_name = add_img_to_library(args.add, args.name)

    if(len(img_name) == 0):
        print('Input image path as follows: --input YOUR_PATH_HERE.png')
    else:    
        process_img(img_name, args)
