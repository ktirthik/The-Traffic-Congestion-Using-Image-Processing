# import the necessary packages
#opencv 2.4.13
import argparse
import datetime
import time
import cv2
import urllib
import numpy as np
import os
import time
import sys
import subprocess

backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
kk=0
height=400
width=500
cnt=0;
time.sleep(1);
webcam = cv2.VideoCapture(0)  ### for live change the id to be zero
xchg=0
firstFrame = None
ttf=0
centerprevX=0
secondsfin=0
while True:

        startt = time.time()
        fr,frame=webcam.read();
        text = "Unoccupied"

        # resize the frame, convert it to grayscale, and blur it
##	frame = cv2.resize(frame,(960, 540), interpolation = cv2.INTER_CUBIC)
##        print np.shape(frame)
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray1, (21, 21), 0)
##        gray=gray1[100:,120:320];
##        frame=frame[100:,120:320];
        # if the first frame is None, initialize it
        if firstFrame is None:
                firstFrame = gray
                backgroundSubtractor.apply(firstFrame, learningRate=1)
                continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = backgroundSubtractor.apply(gray, learningRate=0)
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 80, 255, cv2.THRESH_BINARY)[1]

                      # show the frame and record if the user presses a key
        cv2.putText(frame, "Status: {}".format(secondsfin), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 1)
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
                break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
