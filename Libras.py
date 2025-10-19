#bibliotecas
import cv2 as cv
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from collections import deque

from joblib import load
clf = load("Amostras.joblib")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)
    base = pts[0]
    pts -= base
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

camera = cv.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    smooth = deque(maxlen=7)  # suavização
    while True:
        ok, frame = cap.read()
        if not ok: break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)

        label = ""
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            feats = normalize_landmarks(hand.landmark).reshape(1, -1)
            pred = clf.predict(feats)[0]
            smooth.append(pred)
            # maioria
            vals, counts = np.unique(smooth, return_counts=True)
            label = vals[np.argmax(counts)]

            # desenhar
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesto: {label}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("LIBRAS - demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pra sair
            break

cap.release()
cv2.destroyAllWindows()

