import glob
import os
import cv2
import numpy as np
from radar_utils.filter import radar_interference_filter
from radar_utils.scan import PolarScan, EuclideanScan
from radar_utils.transformations import polarToEuc
import pickle
import argparse

"""
Script to preprocess computationally intensive operations on the dataset.
Currently implemented:
    - Interference reduction filter
    - Cartesian conversion
"""

# Parse input
parser = argparse.ArgumentParser("Cache generator",
                                 "A small script to do CPU-intensive preprocessing and cache the results.")
parser.add_argument("dataset_path")
args = parser.parse_args()
if args.dataset_path is None:
    raise ValueError("You need to specify the dataset path")

# dataset_collection_path = "/work/Data/release/"
# datasets = glob.glob(dataset_collection_path + "*/Radar*/")

datasets = [args.dataset_path]  # List the paths here if you want to do multiple datasets at once
for dataset_path in datasets:
    imgpaths = glob.glob(dataset_path + "/*.bmp")
    imgpaths.sort()

    for imgpath in imgpaths:
        filepath = '/'.join(imgpath.split('/')[:-1])
        filename = imgpath.split('/')[-1]
        interferece_path = f"{filepath}/radar_interference_filtered/"
        interference_path_iter = f"{interferece_path}{filename}"
        cart_path = f"{filepath}/radar_interference_filtered_cartesian/"
        cart_path_iter = cart_path + filename.split('.')[0] + '.pkl'
        if not os.path.exists(interference_path_iter):
            scan_polar = PolarScan.load(imgpath)
            scan_polar = radar_interference_filter(scan_polar)
            scan_polar = radar_interference_filter(scan_polar, 2)

            os.makedirs(interferece_path, exist_ok=True)
            os.makedirs(cart_path, exist_ok=True)
            cv2.imwrite(interference_path_iter, scan_polar.img)

            if not os.path.exists(cart_path_iter):
                scan_euc = polarToEuc(scan_polar, new_size=np.array([4000, 4000]))
                with open(cart_path_iter, 'wb') as f:
                    pickle.dump(scan_euc, f)
                print(f"Wrote out both for {imgpath}")
