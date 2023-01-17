import cv2 as cv
import numpy as np
import time
import os
import hand_tracking_module as htm

brush_thickness = 15
eraser_thickness = 50

folder_path = 'header'
my_list = os.listdir(folder_path)
my_list = sorted(my_list)
# print(my_list)

overlay_list = []

for image_path in my_list:
    image = cv.imread(f'{folder_path}/{image_path}')
    overlay_list.append(image)

# print(len(overlay_list))

header = overlay_list[0]
draw_color = (255, 0, 255)

cap = cv.VideoCapture(0)
width = 740
height = 480
dim = (width, height)
# height, width , channels = img.shape

detector = htm.HandDetector(detection_con=0.85)
xp, yp = 0, 0
img_canvas = np.zeros((height, width, 3), np.uint8)
while True:
    # 1. import image
    success, img = cap.read()
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    img = cv.flip(img, 1)

    # 2. find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list)
        # tip of index and middle finger
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        # 3. check which fingers are up
        fingers = detector.fingers_up()

        # 4. if selection mode - two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('selection mode')
            # checking for the click
            if y1 < 88:
                if 0 < x1 < 250:
                    header = overlay_list[0]
                    draw_color = (255, 0, 255)
                elif 251 < x1 < 380:
                    header = overlay_list[1]
                    draw_color = (255, 0, 0)
                elif 400 < x1 < 480:
                    header = overlay_list[2]
                    draw_color = (0, 255, 0)
                elif 550 < x1 < 740:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)
            cv.rectangle(img, (x1, y1-25), (x2, y2+25), draw_color, cv.FILLED)

        # 5. if drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, draw_color, cv.FILLED)
            print('drawing mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv.line(img_canvas, (xp, yp), (x1, y1),
                        draw_color, eraser_thickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv.line(img_canvas, (xp, yp), (x1, y1),
                        draw_color, brush_thickness)
            xp, yp = x1, y1

    img_gray = cv.cvtColor(img_canvas, cv.COLOR_BGR2GRAY)
    _, img_inv = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY_INV)
    img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, img_inv)
    img = cv.bitwise_or(img, img_canvas)
    # setting the header image
    img[0:88, 0:740] = header
    # img = cv.addWeighted(img, 0.5, img_canvas, 0.5, 0)
    cv.imshow("image", img)
    cv.imshow("canvas", img_canvas)

    cv.waitKey(1)
