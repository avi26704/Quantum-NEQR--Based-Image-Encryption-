import numpy as np
import math

def mse(img_a, img_b):
    """
    Mean Squared Error between two images.
    img_a, img_b: numpy arrays of shape (4,4,3) in uint8
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    diff2 = (a - b) ** 2
    return float(np.mean(diff2))

def mae(img_a, img_b):
    """
    Mean Absolute Error between two images.
    """
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    diff_abs = np.abs(a - b)
    return float(np.mean(diff_abs))

def psnr(img_a, img_b, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio in dB.
    If images are identical, PSNR = inf.
    """
    m = mse(img_a, img_b)
    if m == 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / m)
