import cv2 as cv
import time
import os
import hand_tracking_module as htm
w_cam, h_cam = 640, 480

cap = cv.VideoCapture(0)

cap.set(3, w_cam)
cap.set(4, h_cam)

folder_path = "../finger_contours"
my_list = os.listdir(folder_path)
my_list = sorted(my_list)
over_lay_list = []

for img_path in my_list:
    image = cv.imread(f'{folder_path}/{img_path}')
    over_lay_list.append(image)

p_time = 0

detector = htm.HandDetector(detection_con=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        fingers = []
        # thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)

        h, w, c = over_lay_list[total_fingers-1].shape
        img[0:h, 0:w] = over_lay_list[total_fingers-1]

        cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(total_fingers), (45, 375),
                   cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv.putText(img, f'FPS: {int(fps)}', (400, 70),
               cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("img", img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
