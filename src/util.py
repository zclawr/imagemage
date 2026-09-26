from PIL import Image
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
import os
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io
import hashlib
from pillow_heif import register_heif_opener

# Register HEIF opener with Pillow
register_heif_opener()

HASH_LENGTH = 12

history_path = os.path.join(os.path.dirname(os.getcwd()), 'history.txt')

supported_extensions = [
    'wav',
    'png',
    'jpg',
    'jpeg'
]
def get_last_output():
    with open(history_path, "r", encoding="utf-8") as file:
            path = file.read()
    return path

def write_last_output(path):
    with open(history_path, "w") as file:
        file.write(path)

def load_input_file(path):
    extension = path.split(".")[-1].strip()
    sr = None
    is_sound = False
    if extension == 'wav':
        image, sr = sound_to_img(path)
        is_sound = True
    elif extension == 'mat':
        image = mat_to_img(path)
        image = preprocess_img(np.array(image, dtype=np.uint8))
    else:
        image = Image.open(path)
        image = preprocess_img(np.array(image, dtype=np.uint8))
    return image, is_sound, sr

def save_output(cmd, dir, img, is_sound, sr, idx=-1):
    file_name = short_deterministic_hash(cmd, HASH_LENGTH)
    if idx != -1:
        file_name += f'_{idx}'
    if not is_sound:
        path = os.path.join(dir, f'{file_name}.png')
        Image.fromarray(np.array(img, dtype=np.uint8)).save(path)
    else:
        path = os.path.join(dir, f'{file_name}.wav')
        audio = img_to_sound(img, sr)
        wavfile.write(path, sr, np.array(audio, dtype=np.int32))
    print(f'Saved {path}')

def mat_to_img(path):
    mat = scipy.io.loadmat(path)
    print(mat.keys())
    arr = mat["Problem"][0,0][1].toarray() 
    arr_normalized = (arr - np.min(arr)) / np.max(arr)
    img = np.array(np.floor(arr_normalized * 255), dtype=np.uint8)
    return img

def img_to_sound(img, sr):
    # Collapse to 1 channel
    img = img[:,:,0]
    t, waveform = signal.istft(img, fs=sr)
    return waveform

def sound_to_img(path):
    sampling_rate, data = wavfile.read(path)
    frequencies, times, stft_matrix = signal.stft(data, fs=sampling_rate)
    # Extend to 3 channels for RGB
    stft_matrix = np.repeat(stft_matrix[:,:,np.newaxis], 3, axis=-1)
    return stft_matrix, sampling_rate

def format_num(n, digits=3):
    for i in range(1, digits+1):
        if n < 10 ** i:
            return ('0' * (digits - i)) + f'{n}'
    return f"{n}"

def preprocess_img(img):
    # If only 1-channel, expand img to 3 channels (RGB)
    if len(img.shape) == 2:
        img = np.repeat(img[:,:, np.newaxis], 3, axis=-1)
    # Remove alpha (opacity) channel
    if img.shape[2] == 4:
        img = img[:,:,:-1]
    # Ensure even width/height, which is necessary for some functions
    if img.shape[0] % 2 == 1:
        img = img[:-1, :, :]
    if img.shape[1] % 2 == 1:
        img = img[:, :-1, :]
    return img

def short_deterministic_hash(input_string, length = 8):
    full_hash = hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    return full_hash[:length]