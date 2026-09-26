import numpy as np

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
    pad_img = np.zeros(shape=(img.shape[0] + 2, img.shape[1] + 2))
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
    sobel_kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_2 = np.transpose(sobel_kernel_1)
    for i in range(pad_img.shape[0] - 2):
        for j in range(pad_img.shape[1] - 2):
            local = pad_img[i:i+3, j:j+3]
            conv_1 = np.sum(np.sum(local * sobel_kernel_1, axis=0), axis=0)
            conv_2 = np.sum(np.sum(local * sobel_kernel_2, axis=0), axis=0)
            mg_img[i,j] = np.sqrt((conv_1 ** 2) + (conv_2 ** 2))
    out[:,:,k] = mg_img
  return out


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

  return out
    
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

    return out

def gaussian_blur(img_np, dirty=1):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1/16)
    if dirty==1:
        kernel += np.random.uniform(low=0.1, high=0.1, size=kernel.shape)
    return convolve(img_np, kernel)

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
  return out