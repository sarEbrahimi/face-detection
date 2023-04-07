import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import chisquare , itemfreq

def calc_lbp(img):
    # calculate lbp
    image = local_binary_pattern(img, 16, 8, method='uniform')
    hist =cv2.calcHist([np.float32(image)] , [0],None,[256],[0,256])
    #chi = chisquare(image)

    return hist


