import numpy as np


def prune_weak_returns(scan_rn, wave_noise_threshold):
    min_return = wave_noise_threshold
    scale = 255 / (255 - min_return)
    level = np.ones_like(scan_rn.img, dtype='int16') * min_return
    diff = scan_rn.img - level
    diff = np.clip(diff, 0, None)
    diff = (diff * scale).astype('uint8')
    scan_rn.img = diff
    return scan_rn
