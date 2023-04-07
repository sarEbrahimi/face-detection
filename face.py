"""
first is face detection
then face recognition

list of face recognition algorithms:
1. LBPH
2.Eigenfaces
3.Fisherfaces
4.SIFT
5.SURF
"""
# viola jones haar cascade classifier
import cv2
from findface import detectAndDisplay
from lbph import calc_lbp
from trainedRec import show_data
from facerecognition import comp_his

# 1.read data file
dataset = []
dataset = show_data()

# 2.we need their local binary pattern <in lbp function proccess histogram in the end>
lbp_hist = []
for data in dataset:
    lbp_hist.append( calc_lbp(data) )

# 3.what user entered to be proccessed?
img = cv2.imread('1.png')
flag , image = detectAndDisplay(img)

# 4.how we know that is not a joke? < IF THERE IS FACE DO REST >
if flag==False:
    raise Exception("Sorry there is not face in your image")

cv2.imshow('capture__Face Detection', image)
cv2.waitKey()

# 5.we need user_enter_image local binary pattern
main_hist = calc_lbp(image)

# 6.OK, we have calc_hist and main_hist. It is time to compare them.
cmp_hist = []
for hist in lbp_hist:
    cmp_hist.append( comp_his( lbp_hist, main_hist ) )

# 7.NOW, sort cmp_hist