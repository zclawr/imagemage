import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def ecb(img_np):
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
    data = img_np.tobytes()
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(data)
    # Repack to same datatype and shape
    cipher_img = np.frombuffer(ciphertext)
    cipher_img = cipher_img.reshape(img_np.shape)

    return cipher_img
