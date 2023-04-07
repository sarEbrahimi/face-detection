"""
algotithm runs LBPH on the images and stores the trained classifier here
"""
import cv2
import os
import glob
import lbph
from facerecognition import comp_his

# 1.read data
def show_data():
    # read dataset
    dataset = []
    data_dir = 'att/'
    data_path = os.path.join(data_dir,'*png' )
    files = glob.glob(data_path)
    for f in files:
        img = cv2.read(f , cv2.IMREAD_GRAYSCALE)
        dataset.append(img)
    return dataset