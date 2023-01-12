import cv2 as cv
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


cap = cv.VideoCapture('../videos/1.mp4')
p_time = 0
while True:
    success, img = cap.read()
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx, cy), 3, (255, 0, 2), cv.FILLED)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv.putText(img, str(int(fps)), (70, 50),
               cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("image", img)
    cv.waitKey(1)
