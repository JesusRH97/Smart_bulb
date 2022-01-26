import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm

wCam, hCam = 640, 480

pTime = 0
cTime = 0
cont = 0
average = np.zeros((10, ))

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)

    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    cTime = time.time()
    #print(cTime - pTime)
    pTime = cTime

    if len(lmList):

        #Big finger
        cv2.circle(img, (lmList[4][1], lmList[4][2]), 15, (255, 0, 255), cv2.FILLED)
        big_finger = math.hypot(lmList[4][1] - lmList[0][1], lmList[4][2] - lmList[0][2])

        #Index finger
        cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (255, 0, 255), cv2.FILLED)
        index_finger = math.hypot(lmList[8][1] - lmList[0][1], lmList[8][2] - lmList[0][2])

        #Line between both fingers
        cv2.line(img, (lmList[4][1], lmList[4][2]), (lmList[8][1], lmList[8][2]), (255, 0, 255), 3)

        distance_between_fingers = (math.hypot(lmList[4][1] - lmList[8][1], (lmList[4][2] - lmList[8][2]))) / (index_finger/17.0)

        average[cont] = distance_between_fingers
        cont += 1

        if cont == 10:
            print(np.mean(average))
            cont = 0


    cv2.imshow("Img", img)
    cv2.waitKey(1)