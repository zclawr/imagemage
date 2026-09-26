import numpy as np

def bitslice(img_np, idx=None):
    v_bin = np.vectorize(np.binary_repr)
    bits = v_bin(img_np, width=8)
    outs = np.zeros(shape=(8, img_np.shape[0], img_np.shape[1], img_np.shape[2]))
    for i in range(outs.shape[1]):
        for j in range(outs.shape[2]):
            for k in range(outs.shape[3]):
                bitstr = bits[i,j,k]
                for n in range(8):
                    outs[n,i,j,k] = int(bitstr[n]) * np.uint8(255)
    if idx==None:
        return np.array(outs, dtype=np.uint8)
    else:
        return np.array(outs[idx], dtype=np.uint8)