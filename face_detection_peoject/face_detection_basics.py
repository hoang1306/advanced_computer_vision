import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)
# cap = cv.VideoCapture("../videos/1.mp4")
p_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_c = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bbox_c.xmin*iw), int(bbox_c.ymin*ih), \
                int(bbox_c.width*iw), int(bbox_c.height*ih)
            cv.rectangle(img, bbox, (255, 0, 0), 2)
            cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv.putText(img, f'FPS: {int(fps)}', (20, 70),
               cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv.imshow("img", img)

    cv.waitKey(1)
