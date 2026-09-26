import numpy as np

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