import cv2 as cv
import time
import hand_tracking_module as htm

p_time = 0
c_time = 0
cap = cv.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    # img = detector.find_hands(img, False)
    lm_list = detector.find_position(img)
    # lm_list = detector.find_position(img, draw=False)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
    cv.imshow("image", img)
    cv.waitKey(1)
