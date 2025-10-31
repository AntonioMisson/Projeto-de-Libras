import json
import numpy as np
from collections import deque
from pathlib import Path
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# Hiperparâmetros precisam bater com o treino
INPUT_SIZE = 63
SEQ_LEN = 40
HIDDEN = 128
LAYERS = 2
BIDIR = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MAP = json.loads(Path("class_mapping.json").read_text(encoding="utf-8"))
INV_CLASS = {v: k for k, v in CLASS_MAP.items()}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    base = pts[0].copy()
    pts -= base
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

def pad_or_truncate(seq, target_len):
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T > target_len:
        return seq[-target_len:]  # janela mais recente
    pad = np.zeros((target_len - T, seq.shape[1]), dtype=seq.dtype)
    return np.vstack([pad, seq])

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden, layers, num_classes, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def main():
    # Carregar normalização e modelo
    stats = np.load("norm_stats.npz")
    mean = stats["mean"]  # shape (1, 63)
    std  = stats["std"]

    model = LSTMClassifier(INPUT_SIZE, HIDDEN, LAYERS, num_classes=len(CLASS_MAP), bidir=BIDIR).to(DEVICE)
    model.load_state_dict(torch.load("lstm_model.pt", map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" Webcam não encontrada.")
        return

    seq_buf = deque(maxlen=SEQ_LEN)     # armazena os últimos frames (63,)
    smooth_preds = deque(maxlen=7)      # suavização por maioria

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                feats = normalize_landmarks(hand.landmark)  # (63,)
                seq_buf.append(feats)

            label_txt = ""
            if len(seq_buf) >= 8:  # precisa de alguns frames mínimos
                seq_arr = np.array(seq_buf, dtype=np.float32)  # (t, 63)
                seq_arr = pad_or_truncate(seq_arr, SEQ_LEN)    # (SEQ_LEN, 63)
                seq_arr = (seq_arr - mean) / (std + 1e-6)
                x = torch.from_numpy(seq_arr[None, ...]).float().to(DEVICE)  # (1, T, 63)
                with torch.no_grad():
                    logits = model(x)
                    pred_id = int(logits.argmax(1).item())
                smooth_preds.append(pred_id)

                # maioria
                vals, counts = np.unique(np.array(smooth_preds), return_counts=True)
                pred_id = int(vals[np.argmax(counts)])
                label_txt = INV_CLASS[pred_id]

            cv2.putText(frame, f"Gesto: {label_txt}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label_txt else (200, 200, 200), 2)
            cv2.imshow("Inferencia LSTM - LIBRAS", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
