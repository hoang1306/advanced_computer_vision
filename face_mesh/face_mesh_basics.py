import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture(0)
p_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
# max_num_faces=2 => number person
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms, mp_face_mesh.FACEMESH_CONTOURS,
                                   draw_spec, draw_spec)
            for id, lm in enumerate(face_lms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv.putText(img, f'FPS: {int(fps)}', (20, 70),
               cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.imshow("image", img)

    cv.waitKey(1)
