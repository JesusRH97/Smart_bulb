import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while(True):

    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str("fps = "+str(int(fps))), (10, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    cv2.imshow("Imagen", img)
    cv2.waitKey(1)