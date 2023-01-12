import cv2 as cv
import numpy as np
import time
import pose_module as pm


cap = cv.VideoCapture(0)
detector = pm.PoseDetector()
count = 0
dir = 0
p_time = 0
while True:
    success, img = cap.read()
    img = cv.resize(img, (1280, 840))
    # img = cv.imread("ai_personal_trainer/test.jpg")

    img = detector.find_pose(img, False)
    lm_list = detector.find_position(img, False)
    if len(lm_list) != 0:
        # right arm
        # detector.find_angle(img, 12, 14, 16)

        # left arm
        angle = detector.find_angle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        # check for the dumbbell curls
        color = (255, 0, 255)

        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # draw bar
        cv.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv.rectangle(img, (1100, int(bar)), (1175, 650),
                     (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{int(per)} %', (1100, 75),
                   cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # draw curl count
        cv.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)
        cv.putText(img, f'{str(int(count))}', (45, 670),
                   cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv.putText(img, f'{str(int(fps))}', (50, 100),
               cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv.imshow("image", img)
    cv.waitKey(1)
