import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
p_time = 0
c_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
                cv.circle(img, (cx, cy), 6, (255//(id+1), id, id*8), cv.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv.imshow("image", img)
    cv.waitKey(1)
