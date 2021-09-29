#opencv 2.4.13
import argparse
import datetime
import time
import cv2
import urllib
import numpy as np
import os
import serial


MIN_MATCH_COUNT=40
lt=1;
gt=1;
detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
cntnew1='11'
cntnew2='22'
imgNamenew1=str(cntnew1)+'.jpg';
imgNamenew2=str(cntnew2)+'.jpg';
trainImg=cv2.imread(imgNamenew1,0)
trainImg2=cv2.imread(imgNamenew2,0)
trainKP1,trainDesc1=detector.detectAndCompute(trainImg,None)
trainKP2,trainDesc2=detector.detectAndCompute(trainImg2,None)
kk=0
height=400
width=500
cnt=0;
##time.sleep(5);
webcam1 = cv2.VideoCapture(0)
firstFrame = None
##ser = serial.Serial('COM7', 9600)#Communication Baudrate
data=['C','D'];
val=5
backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
amb=0
chkval=1
val=2
##cntnam1=0
##cntnam2=1
while True:
##        imgPath=urllib.urlopen(url)
##        imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
##        frame=cv2.imdecode(imgNp,-1)
        cntcam1=0
        cntcam2=0
        amb=0
        for cm in range(1,2):
                if cm==1:
                        fr,frame1=webcam1.read();
                        frame=frame1;
                elif cm==2:
                        fr,frame2=webcam2.read();
                        frame=frame2;
                text = "Unoccupied"

                # resize the frame, convert it to grayscale, and blur it
        ##	frame = cv2.resize(frame,(960, 540), interpolation = cv2.INTER_CUBIC)
##                print frame.shape
                gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
                gray=gray1;
                frame=frame;
                if firstFrame is None:
                        firstFrame = gray
                        backgroundSubtractor.apply(firstFrame, learningRate=1)
                        continue

                # compute the absolute difference between the current frame and
                # first frame
                frameDelta = backgroundSubtractor.apply(gray, learningRate=0)

                # dilate the thresholded image to fill in holes, then find contours
                # on thresholded image
                thresh = cv2.erode(frameDelta, None, iterations=2)
                thresh1=thresh;
                cnts = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                # loop over the contours
                cnt=0;
                for c in cnts:
                        (x, y, w, h) = cv2.boundingRect(c)
                        img=frame;
                        if w*h>1000:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cnt=cnt+1;
##                                print cnt
                                imgName=str(cnt)+'.jpg';
                                text = cnt
                                ImageCropped=img[y:y+h,x:x+w];
                                if cm==1:
                                        print ('ok')
                                        cntcam1=cnt;
                                elif cm==2:
                                        cntcam2=cnt;
                                if kk>2:
                                        QueryImgBGR=frame;
                                        QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
                                        queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
                                        matches1=flann.knnMatch(queryDesc,trainDesc1,k=2)
                                        matches2=flann.knnMatch(queryDesc,trainDesc2,k=2)
                                        for i in range(1,2):
                                                if i==1:
                                                        matches=matches1;
                                                        trainKP=trainKP1;
                                                        trainDesc=trainDesc1;
                                                else:
                                                        matches=matches2;
                                                        trainDesc=trainDesc2;
                                                goodMatch=[]
                                                for m,n in matches:
                                                        if(m.distance<0.75*n.distance):
                                                            goodMatch.append(m)
                                                if(len(goodMatch)>MIN_MATCH_COUNT):
                                                        tp=[]
                                                        qp=[]
                                                        for m in goodMatch:
                                                                tp.append(trainKP[m.trainIdx].pt)
                                                                qp.append(queryKP[m.queryIdx].pt)
                                                        tp,qp=np.float32((tp,qp))
                                                        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                                                        h,w=trainImg.shape
                                                        if cm==1:
                                                                cv2.putText(frame1, "############ AMBULANCE DETECTED cam1 ###########!", (h, w),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                                                print ("############ AMBULANCE DETECTED ###########!")
                                                                amb=1
                                                        elif cm==2:
                                                                cv2.putText(frame2, "############ AMBULANCE DETECTED cam2 ###########", (h, w),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)                
                                                                print ("############ AMBULANCE DETECTED ###########!")
                                                                amb=2
                                                print ("")
                        kk=kk+1
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # show the frame and record if the user presses a key
                cv2.imshow("camera1", frame1)
##                cv2.imshow("camera2", frame2)
##                cv2.imshow("Thresh", thresh)
##                cv2.imshow("Frame Delta", frameDelta)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                        break
        value11=''
        value12=''
        value13=''
        value14=''
        value15=''
        value16=''


        if amb==1:
                chkval=1
                val=10   ## time limit setting for ambulance
                print ('ambulance lane A')
                value13=('ambulance lane A')
        elif amb==2:
                chkval=2
                val=10   ## time limit setting for ambulance
                print ('ambulance lane B')
                value14=('ambulance lane A')
        else:
                if cntcam1>cntcam2:
                    chkval=1
                    val=10;  ## density time limit
                    print ('camera 1 cont:::'+str(cntcam1))
                    value11=('camera 1 cont:::'+str(cntcam1))
                if cntcam1<cntcam2:
                    chkval=2  
                    print ('camera 2 cont:::'+str(cntcam2))
                    val=10;## density time limit
                    value12=('camera 1 cont:::'+str(cntcam1))
        if chkval==1:
##                ser.write("C")
                print('side A turned ON'+str(val))
                print('##################')
                value15='side A turned ON'+str(val)
                time.sleep(val)
                chkval=2
                
                
        elif chkval==2:
##                ser.write("D")
                print('side B turned ON'+str(val))
                print('##################')
                value16='side B turned ON'+str(val)
                time.sleep(val)
                chkval=1
        val=2    ## normal state time limit
        value=value11+value12+value13+value14+value15+value16
        dat='BAD WORDS DETECTED'
        f = open('log1.txt','w')
        f.write(value)
        f.close        
# cleanup the camera and close any open windows
webcam1.release()
webcam2.release()


cv2.destroyAllWindows()
