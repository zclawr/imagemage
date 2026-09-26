import numpy as np
import pywt

def low_rank_approximation_svd(img_np, iters):
    """
    Computes the best rank-k approximation of a matrix A using SVD.
    """
    # Perform SVD over each RGB channel separately
    channels = img_np.shape[-1]
    svd_img = np.zeros(shape=img_np.shape, dtype=img_np.dtype)
    svd_img = np.repeat(svd_img[np.newaxis,:,:,:], axis=0, repeats=iters)
    for j in range(iters):
        k = 2 ** j
        for i in range(channels):
            A = img_np[:,:,i]
            U, s, Vt = np.linalg.svd(A, full_matrices=False)

            # Truncate to the first k singular values and vectors
            U_k = U[:, :k]
            s_k = np.diag(s[:k])
            Vt_k = Vt[:k, :]

            # Reconstruct the rank-k approximation
            A_k = U_k @ s_k @ Vt_k
            svd_img[j,:,:,i] = A_k
    return np.array(svd_img, dtype=img_np.dtype)

def low_rank_approximation_svd_per_channel(img_np, k_r, k_g, k_b):
    """
    Computes the best rank-k approximation of a matrix A using SVD.
    """
    # Perform SVD over each RGB channel separately
    channels = img_np.shape[-1]
    svd_img = np.zeros_like(img_np)
    ks = [k_r, k_g, k_b]
    for i in range(channels):
        A = img_np[:,:,i]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # Truncate to the first k singular values and vectors
        U_k = U[:, :ks[i]]
        s_k = np.diag(s[:ks[i]])
        Vt_k = Vt[:ks[i], :]

        # Reconstruct the rank-k approximation
        A_k = U_k @ s_k @ Vt_k
        svd_img[:,:,i] = A_k
    return svd_img

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
    cA_out = np.transpose(np.array(cA_out), (1, 2, 0))
    cH_out = np.transpose(np.array(cH_out), (1, 2, 0))
    cV_out = np.transpose(np.array(cV_out), (1, 2, 0))
    cD_out = np.transpose(np.array(cD_out), (1, 2, 0))
    return cA_out, cH_out, cV_out, cD_out