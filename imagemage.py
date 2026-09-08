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
import subprocess
import pywt
import librosa
import soundfile as sf

# Define constant pathing
root = os.path.dirname(__file__)
img_lib = os.path.join(root, 'img')

def get_DCT_transformation_matrix(M):
  """
  M: The number of basis vectors (integer)

  return: 
  A:  DCT transform matrix (M,M)
  """
  A = np.zeros(shape=(M,M))
  for u in range(M):
    for x in range(M):
      alpha = np.sqrt(2 / M)
      if u == 0:
        alpha = np.sqrt(1 / M)

      A[u,x] = alpha * np.cos((((2 * x) + 1) * u * np.pi) / (2 * M))

  return A

def format_num(n, digits=3):
    for i in range(1, digits+1):
        if n < 10 ** i:
            return ('0' * (digits - i)) + f'{n}'
    return f"{n}"

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
    if img_np.shape[0] % 4 != 0:
        img_np = img_np[:-2,:,:]
    if img_np.shape[1] % 4 != 0:
        img_np = img_np[:,:-2,:]
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

def gradient_magnitude(img_np):
  """
  img: input image (H,W)

  returns:
  mg_img: image after applying magnitude of gradient (H,W)
  """
  out = np.zeros(shape = img_np.shape)
  for k in range(img_np.shape[2]):
    img = img_np[:,:,k]
    # Apply replicate padding to img
    pad_img = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2), dtype=np.int32)
    pad_img[1:-1, 1:-1] = img
    # Top left corner
    pad_img[0,0] = img[0,0]
    # Top right corner
    pad_img[0,-1] = img[0,-1]
    # Bottom left corner
    pad_img[-1,0] = img[-1,0]
    # Bottom right corner
    pad_img[-1,-1] = img[-1,-1]
    # Horizontal padding
    for i in range(0, img.shape[0]):
        pad_img[i+1, 0] = img[i, 0]
        pad_img[i+1, -1] = img[i, -1]
    # Vertical padding
    for i in range(0, img.shape[1]):
        pad_img[0, i+1] = img[0, i]
        pad_img[-1, i+1] = img[-1, i]

    # Apply convolution
    mg_img = np.zeros(shape=img.shape)
    sobel_kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32)
    sobel_kernel_2 = np.transpose(sobel_kernel_1)
    for i in range(pad_img.shape[0] - 2):
        for j in range(pad_img.shape[1] - 2):
            local = pad_img[i:i+3, j:j+3]
            conv_1 = np.sum(np.sum(local * sobel_kernel_1, axis=0), axis=0)
            conv_2 = np.sum(np.sum(local * sobel_kernel_2, axis=0), axis=0)
            mg_img[i,j] = np.sqrt((conv_1 ** 2) + (conv_2 ** 2))
    out[:,:,k] = mg_img
  return np.array(out, dtype=np.uint8)

def bitslice(img_np):
    v_bin = np.vectorize(np.binary_repr)
    bits = v_bin(img_np, width=8)
    outs = np.zeros(shape=(8, img_np.shape[0], img_np.shape[1], img_np.shape[2]))
    for i in range(outs.shape[1]):
        for j in range(outs.shape[2]):
            for k in range(outs.shape[3]):
                bitstr = bits[i,j,k]
                for n in range(8):
                    outs[n,i,j,k] = int(bitstr[n]) * 255
    return np.array(outs, dtype=np.uint8)

def ideal_lpf(img_np, radius):
  """
  img: input image (H,W)
  radius: Radius for the ideal lowpass filter 

  returns:
  out: output image after low-pass filter has been applied in the frequency domain
  """
  out = np.zeros_like(img_np)
  for k in range(img_np.shape[2]):
    img = img_np[:,:,k]
    H = img.shape[0]
    W = img.shape[1]
    kH = 2 * np.ceil(radius)
    kW = kH

    P = H + kH - 1
    Q = W + kW - 1

    img_padded = np.zeros(shape=(P,Q))
    img_padded[:H, :W] = img
    
    lpf = np.zeros(shape=(P,Q))
    for i in range(P):
        for j in range(Q):
            dist = np.sqrt(((i - (P / 2)) ** 2) + ((j - (Q / 2)) ** 2))
            if dist <= radius:
                lpf[i,j] = 1

    dft_img = np.fft.fft2(img_padded)
    dft_img = np.fft.fftshift(dft_img)

    prod = dft_img * lpf
    prod = np.fft.ifftshift(prod)
    ift = np.real(np.fft.ifft2(prod))

    out[:,:,k] = ift[:H, :W]

  return np.array(out, dtype=np.uint8)
    
def convolve(img_np, kernel):
    """
    img: input image (H, W)
    kernel: filter (kH, kW)

    returns:
    f_img: image filtered in frequency domain (H, W)
    """
    out = np.zeros_like(img_np)
    for i in range(img_np.shape[2]):
        img = img_np[:,:,i]
        H = img.shape[0]
        W = img.shape[1]
        kH = kernel.shape[0]
        kW = kernel.shape[1]

        P = H + kH - 1
        Q = W + kW - 1

        img_padded = np.zeros(shape=(P,Q))
        img_padded[:H, :W] = img

        kernel_padded = np.zeros(shape=(P,Q))
        kernel_padded[:kH, :kW] = kernel

        dft_img = np.fft.fft2(img_padded)
        dft_kernel = np.fft.fft2(kernel_padded)

        prod = dft_img * dft_kernel
        f_img = np.real(np.fft.ifft2(prod))

        out[:,:,i] = f_img[1:H+1,1:W+1]

    return np.array(out, dtype=np.uint8)

def gaussian_blur(img_np, dirty=True):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1/16)
    if dirty:
        kernel += np.random.uniform(low=0.1, high=0.1, size=kernel.shape)
    return convolve(img_np, kernel)

def modulo(img_np, mod, channel_linked=True):
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            if channel_linked:
                if(img_np[i,j,0] >= mod and img_np[i,j,1] >= mod and img_np[i,j,2] >= mod):
                    img_np[i,j,:] = img_np[i,j,:] % mod
            else:
                for k in range(img_np.shape[2]):
                    img_np[i,j,k] = img_np[i,j,k] % mod
    return np.array(img_np, dtype=np.uint8)

def adaptive_local_noise_reduction(img_np, var, window_size, tear=True):
  """
  noisy_img: image with noise added (H,W)
  var: overall noise variance
  window_size: filter window (n,n)

  returns: 
  fhat_img: estimate of the image after noise removal (H,W) as a numpy ndarray
  """
  out = np.zeros_like(img_np)
  for i in range(img_np.shape[2]):
    noisy_img = img_np[:,:,i]
    H,W = noisy_img.shape
    fhat_img = np.zeros_like(noisy_img)
    n, _ = window_size 
    m = n // 2
    for x in range(H):
        for y in range(W):
            filtered = []
            for i in range(-m, m):
                for j in range(-m, m):
                    if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                        continue
                    filtered.append(noisy_img[x + i, y + j])
            filtered = np.array(filtered)
            zhat = np.mean(filtered)
            filter_var = np.var(filtered)
            if not tear:
                fhat_img[x,y] = noisy_img[x,y] - min(1, var / filter_var) * (noisy_img[x,y] - zhat)
            else:
                fhat_img[x,y] = noisy_img[x,y] - (var / filter_var) * (noisy_img[x,y] - zhat)
    out[:,:,i] = fhat_img
  return np.array(fhat_img, dtype=np.uint8)

def wavelet(img_np, level):
    cA_out = []
    cH_out = []
    cV_out = []
    cD_out = []
    for c in range(img_np.shape[2]):
        current = img_np[:,:,c]
        for i in range(level):
            cA, (cH, cV, cD) = pywt.dwt2(current, 'haar')
            current = cA
        cA_out.append(cA)
        cH_out.append(cH)
        cV_out.append(cV)
        cD_out.append(cD)
    cA_out = np.transpose(np.array(cA_out, dtype=np.uint8), (1, 2, 0))
    cH_out = np.transpose(np.array(cH_out, dtype=np.uint8), (1, 2, 0))
    cV_out = np.transpose(np.array(cV_out, dtype=np.uint8), (1, 2, 0))
    cD_out = np.transpose(np.array(cD_out, dtype=np.uint8), (1, 2, 0))
    return cA_out, cH_out, cV_out, cD_out

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
                flipped_block = invert(img_np[low_w : high_w, low_h : high_h, :])
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
                flipped_block = ideal_lpf(img_np[low_w : high_w, low_h : high_h, :], 20)
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
                flipped_block, _ = rotate_channels(img_np[low_w : high_w, low_h : high_h, :])
                img_np[low_w : high_w, low_h : high_h, :] = flipped_block
    return img_np

def img_to_sound(img_np):
    audio_signal = librosa.griffinlim(img_np, hop_length=512, n_iter=32)
    return audio_signal

def sound_to_img(img_np):
    y, sr = librosa.load('your_file.wav', sr=None)
    stft = librosa.stft(y, hop_length=512, n_iter=32)
    return stft, sr

def lossy_dct_transform(img_np, M, p):
    """
    img: input image (N,N)
    M: Block size (integer)
    p: percent of smallest DCT coefficients that need to be set to zero (floating point between 0 and 1)

    returns: 
    out: output image after processing (N,N)
    """
    if img_np.shape[0] % M != 0:
        img_np = img_np[:-(img_np.shape[0] % M), :, :]
    if img_np.shape[1] % M != 0:
        img_np = img_np[:, :-(img_np.shape[1] % M), :]

    outs = np.zeros_like(img_np)
    to_zero = int(p * M * M)
    A = get_DCT_transformation_matrix(M)
    for k in range(img_np.shape[2]):
        img = img_np[:,:,k]
        H,W = img.shape
        out = np.zeros_like(img)

        for i in range(0, H, M):
            for j in range(0, W, M):
                block = img[i:i+M, j:j+M]
                T = A @ block @ np.transpose(A)
                if to_zero > 0:
                    T_1dim = np.reshape(T, shape=(T.shape[0] * T.shape[1]))
                    idxs_to_zero = np.argsort(np.abs(T_1dim))[:to_zero]
                    T_1dim[idxs_to_zero] = 0
                    T = np.reshape(T_1dim, shape=(T.shape[0], T.shape[1]))

                F_prime = np.transpose(A) @ T @ A
                out[i:i+M, j:j+M] = F_prime

        outs[:,:,k] = np.maximum(out, 0)
        outs[:,:,k] = np.minimum(out, 255)
    return outs.astype(np.uint8)

def process_img(img_name, args, is_sound=False):
    img = img_to_numpy(img_name)
    # Remove alpha channel
    if len(img.shape) == 2:
        img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:,:,:-1]
    print(f'Image shape (Width, Height, Channels (RGB)): {img.shape}')
    if img.shape[0] % 2 == 1:
        img = img[:-1, :, :]
    if img.shape[1] % 2 == 1:
        img = img[:, :-1, :]
    if args.ecb:
        ecb, d_ecb = ecb_img(img)
        save_transformed_img('ecb', img_name, ecb)
    if args.svd:
        index = 0
        for k_r in range(8):
            for k_g in range(8):
                for k_b in range(8):
                    svd_k = low_rank_approximation_svd_img_per_channel(img, 2 ** k_r, 2 ** k_g, 2 ** k_b)
                    print(svd_k.shape)
                    if args.svd_verbose:
                        svd_name = f'svd_r{k_r}_g{k_g}_b{k_b}'
                    else:
                        svd_name = f'svd_{format_num(index)}'
                    if args.svd_mono:
                        svd_k = to_mono(svd_k)
                    save_transformed_img(svd_name, img_name, svd_k)
                    index += 1
        if args.svd_video:
            dir = os.path.dirname(get_img_path(img_name))
            subprocess.run(["ffmpeg", "-framerate", "25", "-i", f"{dir}/svd_%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{dir}/svd_video.mp4"]) 
    
    if args.svd_uniform:
        for i in range(8):
            svd_k = low_rank_approximation_svd_img_per_channel(img, 2 ** i, 2 ** i, 2 ** i)
            save_transformed_img(f'svd_uniform_{i}', img_name, svd_k)

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
    if args.gradient:
        grad = gradient_magnitude(img)
        save_transformed_img('gradient', img_name, grad)
    if args.bitslice:
        sliced = bitslice(img)
        for i in range(8):
            save_transformed_img(f'bitslice_{i}', img_name, sliced[i,:,:,:])
        if args.bitslice_video:
            dir = os.path.dirname(get_img_path(img_name))
            subprocess.run(["ffmpeg", "-framerate", "10", "-i", f"{dir}/bitslice_%01d.png", "-filter_complex", "[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0", "-loop", "0", f"{dir}/bitslice_video.gif"]) 
    if args.hardlpf:
        radii = [200, 100, 75, 50, 25, 10]
        for i in range(len(radii)):
            blurred = ideal_lpf(img, radii[i])
            save_transformed_img(f'hardlpf_{i}', img_name, blurred)
    if args.gaussblur:
        current = img
        for i in range(5):
            blurred = gaussian_blur(current)
            save_transformed_img(f'gaussblur_{i}', img_name, blurred)
            current = blurred
    if args.modulo:
        mods = [64, 128, 192, 224, 240]
        for i in range(len(mods)):
            result = modulo(img, mods[i])
            save_transformed_img(f'modulo_{i}', img_name, result)
    if args.adaptive:
        var = 0.01
        window_sizes = [(4,4), (9,9), (16,16)]
        for i in range(len(window_sizes)):
            result = adaptive_local_noise_reduction(img, var, window_sizes[i])
            save_transformed_img(f'adaptive_{i}', img_name, result)
    if args.wavelet:
        levels = [1, 2, 3, 4, 5, 6]
        for i in range(len(levels)):
            cA, cH, cV, cD = wavelet(img, levels[i])
            save_transformed_img(f'wavelet_cA_{i}', img_name, cA)
            save_transformed_img(f'wavelet_cH_{i}', img_name, cH)
            save_transformed_img(f'wavelet_cV_{i}', img_name, cV)
            save_transformed_img(f'wavelet_cD_{i}', img_name, cD)
    if args.lossy_dct:
        M = 2
        percents = [0.75, 0.9, 0.975]
        for i in range(len(percents)):
            out = lossy_dct_transform(img, M, percents[i])
            save_transformed_img(f'lossy_dct_{i}', img_name, out)
    if args.lossy_dct_subtract:
        M = 2
        percents = [0.75, 0.9, 0.975]
        for i in range(len(percents)):
            out = lossy_dct_transform(img, M, percents[i])
            out = out - img
            out = np.maximum(out, 0)
            out = np.minimum(out, 255)
            save_transformed_img(f'lossy_dct_sub{i}', img_name, out)
    if args.sharpen:
        current = img
        for i in range(5):
            blurred = gaussian_blur(current, dirty=False)
            diff = np.minimum(np.maximum(blurred - img, 0), 255)
            save_transformed_img(f'sharpen_{i}', img_name, diff)
            current = blurred
    if args.puzzle:
        for i in range(1, 9):
            puzzled = puzzle(img, 2 ** i)
            save_transformed_img(f'puzzle_{i}', img_name, puzzled)
    if args.checker_invert:
        for i in range(1, 9):
            to_modify = np.copy(img)
            checkered = checker_invert(to_modify, 2 ** i)
            save_transformed_img(f'checker_invert_{i}', img_name, checkered)
    if args.checker_hardlpf:
            for i in range(1, 9):
                checkered = checker_hardlpf(img, 2 ** i)
                save_transformed_img(f'checker_hardlpf_{i}', img_name, checkered)
    if args.checker_chrotate:
            for i in range(1, 9):
                to_modify = np.copy(img)
                checkered = checker_chrotate(to_modify, 2 ** i)
                save_transformed_img(f'checker_chrotate_{i}', img_name, checkered)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--add', default='', type=str)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--input', default='', type=str)
    parser.add_argument('--sound', default='', type=str)

    parser.add_argument('--ecb', action="store_true")
    parser.add_argument('--svd', action="store_true")
    parser.add_argument('--svd_uniform', action="store_true")
    parser.add_argument('--mono', action="store_true")
    parser.add_argument('--rcwalk', action="store_true")
    parser.add_argument('--chrotate', action="store_true")
    parser.add_argument('--invert', action="store_true")
    parser.add_argument('--gradient', action="store_true")
    parser.add_argument('--bitslice', action="store_true")
    parser.add_argument('--hardlpf', action="store_true")
    parser.add_argument('--gaussblur', action="store_true")
    parser.add_argument('--modulo', action="store_true")
    parser.add_argument('--adaptive', action="store_true")
    parser.add_argument('--wavelet', action="store_true")
    parser.add_argument('--lossy_dct', action="store_true")
    parser.add_argument('--lossy_dct_subtract', action="store_true")
    parser.add_argument('--sharpen', action="store_true")
    parser.add_argument('--puzzle', action="store_true")
    parser.add_argument('--checker_invert', action="store_true")
    parser.add_argument('--checker_hardlpf', action="store_true")
    parser.add_argument('--checker_chrotate', action="store_true")

    # Advanced options
    parser.add_argument('--svd_verbose', action="store_true")
    parser.add_argument('--svd_mono', action="store_true")
    parser.add_argument('--svd_video', action="store_true")
    parser.add_argument('--bitslice_video', action="store_true")

    args = parser.parse_args()

    is_sound = False

    if args.sound:
        img_name = args.sound
        is_sound = True
    img_name = args.input
    if len(args.add) > 0:
        print(f'Adding {args.name} to library')
        img_name = add_img_to_library(args.add, args.name)

    if(len(img_name) == 0):
        print('Input image path as follows: --input YOUR_PATH_HERE.png')
    else:    
        process_img(img_name, args, is_sound=is_sound)
