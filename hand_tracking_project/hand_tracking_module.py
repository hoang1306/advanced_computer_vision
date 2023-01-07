import cv2 as cv
import mediapipe as mp
# import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        # self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hands_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hands_number]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 6, (255, 0, 0), cv.FILLED)
                # cv.circle(img, (cx, cy), 6, (255//(id+1),id, id*8), cv.FILLED)
        return lm_list
# def main():
#     p_time = 0
#     c_time = 0
#     cap = cv.VideoCapture(0)
#     detector = HandDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.find_hands(img)
#         lm_list = detector.find_position(img)
#         if len(lm_list) != 0:
#             print(lm_list[2])

#         c_time = time.time()
#         fps = 1/(c_time-p_time)
#         p_time = c_time

#         cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3,(255, 0, 255), 3)
#         cv.imshow("image", img)
#         cv.waitKey(1)
# if __name__ == "__main__":
#     main()
