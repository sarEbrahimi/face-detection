# find face in the image
# if there was not any face we shouldnt accept picture

from __future__ import print_function
import cv2

def detectAndDisplay(frame):
    # is there face in the picture or not
    flag = False

    # Reading the image make it to gray
    frame_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #frame_gray = cv2.equalizeHist(frame_gray)
    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(frame_gray,scaleFactor=1.1,minNeighbors=9)

    # Iterating through rectangles of detected faces
    for (x,y,w,h) in faces_rect:
        flag = True
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    return flag,frame

