import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import socket

max_num_hands = 1
geture = {0: 'stop', 1: 'fire'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

total_result = []

# # knn
# df = pd.read_csv('data\hand.csv', header=None)
# x = df.iloc[:, :-1].to_numpy().astype(np.float32)
# y = df.iloc[:, -1].to_numpy().astype(np.float32)

# # cv2의 knn 머신러닝 알고리즘을 사용할 수 있다.
# knn = cv2.ml.KNearest_create()
# knn.train(x, cv2.ml.ROW_SAMPLE, y)


countss = 0


def click(event, x, y, flags, params):
    global data
    global countss
    if event == cv2.EVENT_LBUTTONDOWN:
        total_result.append(data)
        countss += 1
        print(countss)


cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)
with mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:

                joint = np.zeros((21, 3))  # 21개 x,y,z

                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,
                            10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1

                v = v / (np.linalg.norm(v, axis=1))[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[
                                  1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                angle = np.degrees(angle)
                data = np.array([angle], dtype=np.float32)

                # ret, rdata, neig, dist = knn.findNearest(
                #     data, 5)  # 가장 가까운 5개를 기준
                # idx = int(rdata[0][0])
                # print(idx)

                # send_data = str(idx)

                data = np.append(data, 1)
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Dataset', img)
        if cv2.waitKey(1) == ord('q'):
            break

total_result = np.array(total_result, dtype=np.float32)
df = pd.DataFrame(total_result)
df.to_csv('hand3.csv', mode='a', index=None, header=None)
cap.release()
cv2.destroyAllWindows()
