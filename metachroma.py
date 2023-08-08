import cv2
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from queue import Queue

cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
cv2.namedWindow("img", cv2.WINDOW_FREERATIO)
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap=cv2.VideoCapture(0)

cap.set(3,480)
cap.set(4,684)

segmentor=SelfiSegmentation()

buffersize=25*2 #amount of frames to store

q = Queue(maxsize = buffersize)

c=0

sequence=[]

while True:
    ret,img=cap.read()
    if q.qsize()==buffersize:
        imgout=segmentor.removeBG(img,q.get(),threshold=0.8)
    else:
        imgout=segmentor.removeBG(img,(0,255,0),threshold=0.8)
   
    cv2.imshow("img",imgout)

    cv2.waitKey(1)
  
    q.put(imgout)