import cv2 as cv
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode=False, up_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(self.mode, self.up_body, self.smooth, self.detection_con, self.track_con)
        self.pose = self.mp_pose.Pose()

    def find_pose(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        # print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, (255, 0, 2), cv.FILLED)
        return lm_list

# def main():
#     cap = cv.VideoCapture('../videos/1.mp4')
#     p_time = 0
#     detector = PoseDetector()
#     while True:
#         success, img = cap.read()
#         scale_percent = 40
#         width = int(img.shape[1] * scale_percent / 100)
#         height = int(img.shape[0] * scale_percent / 100)
#         dim = (width, height)
#         img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
#         img = detector.find_pose(img)
#         lm_list= detector.find_position(img, draw=False)
#         if len(lm_list) != 0:
#             print(lm_list[14])
#             cv.circle(img, (lm_list[14][1], lm_list[14][2]), 3, (255, 0, 2), cv.FILLED)
#         c_time = time.time()
#         fps = 1/(c_time - p_time)
#         p_time = c_time
#         cv.putText(img, str(int(fps)), (70, 50),
#                 cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#         cv.imshow("image", img)
#         cv.waitKey(1)


# if __name__ == "__main__":
#     main()
