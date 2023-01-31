import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm

wCam, hCam = 640, 480

pTime = 0
cTime = 0
cont = 0
values = np.zeros((10, ))

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector()

while True:
    for i in range(len(values)):

        success, img = cap.read()
        img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)

        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

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

            values[i] = distance_between_fingers

            #print(average[i])

        cv2.imshow("Img", img)
        cv2.waitKey(1)

    print(np.mean(values))

