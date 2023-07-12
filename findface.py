# check whether query comprise face or not

import cv2

def face(query):
    flag = False
    frame_gray = cv2.cvtColor(query,cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(frame_gray,scaleFactor=1.1,minNeighbors=9)
    # Iterating through rectangles of detected faces
    for (x,y,w,h) in faces_rect:
        flag = True
        cv2.rectangle(query,(x,y),(x+w,y+h),(0,255,0),2)
        query = query[y:y+h , x:x+w]
    return flag , query
