import cv2 as cv
# import mediapipe as mp
import pose_module as pm
import time

# cap = cv.VideoCapture('../videos/1.mp4')
cap = cv.VideoCapture(0)
p_time = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    # scale_percent = 40
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    img = detector.find_pose(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list[14])
        cv.circle(img, (lm_list[14][1], lm_list[14][2]),
                  3, (255, 0, 2), cv.FILLED)
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv.putText(img, str(int(fps)), (70, 50),
               cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("image", img)
    cv.waitKey(1)
