import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_mode=False,
                 max_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        # max_num_faces=2 => number person
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_mode,
                                                    self.max_faces,
                                                    self.refine_landmarks,
                                                    self.min_detection_confidence,
                                                    self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(face_lms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # cv.putText(img, str(id), (x, y),
                    #            cv.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    cap = cv.VideoCapture(0)
    p_time = 0
    detector = FaceMeshDetector(max_faces=1)
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img, True)
        # if len(faces) != 0:
        #     print(len(faces))
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        cv.putText(img, f'FPS: {int(fps)}', (20, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        cv.imshow("image", img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
