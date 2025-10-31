import cv2
import os
import time
import json
import numpy as np
import mediapipe as mp
from pathlib import Path


CLASSES = [""]   # adicionar as classes

# Pasta-base onde salvaremos as sequências
DATA_DIR = Path("Dados")

# Opcional: salvar um mapeamento de classes pra usar no treino/inferência
(Path(".") / "class_mapping.json").write_text(json.dumps({c: i for i, c in enumerate(CLASSES)}, ensure_ascii=False, indent=2), encoding="utf-8")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    """
    Recebe uma lista de 21 landmarks (x,y,z em [0..1] relativos à imagem).
    Centraliza no punho (id 0) e escala pela distância do punho ao MCP do dedo médio (id 9).
    Retorna vetor flat (63,).
    """
    pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)
    base = pts[0].copy()
    pts -= base
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()

def main():
    for cls in CLASSES:
        (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

    print("[controles]")
    print(f"- Teclas 1..{len(CLASSES)} para selecionar classe (atual em destaque).")
    print("- R para iniciar/parar gravação de uma sequência.")
    print("- Q ou ESC para sair.\n")

    selected_idx = 0
    recording = False
    buffer_seq = []  # frames da sequência atual
    saved_count = {c: len(list((DATA_DIR / c).glob("*.npy"))) for c in CLASSES}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" Webcam não encontrada.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        last_info_time = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            landmarks_vec = None

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                landmarks_vec = normalize_landmarks(hand.landmark)  # (63,)

            # HUD
            cls = CLASSES[selected_idx]
            status = "REC" if recording else "idle"
            cv2.putText(frame, f"Classe: {cls}  |  Status: {status}  |  Salvos: {saved_count[cls]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0) if recording else (200, 200, 200), 2)

            if recording:
                # Registra mesmo sem mão detectada (opcional); aqui só registra se a mão foi detectada
                if landmarks_vec is not None:
                    buffer_seq.append(landmarks_vec)

                # dica visual piscando
                if (time.time() % 1.0) < 0.5:
                    cv2.circle(frame, (20, 60), 10, (0, 0, 255), -1)

            cv2.imshow("Gravador de sequencias (MediaPipe Hands)", frame)
            k = cv2.waitKey(1) & 0xFF

            # Troca de classe: teclas 1..N
            if ord('1') <= k <= ord(str(min(len(CLASSES), 9))):
                selected_idx = k - ord('1')

            # R -> inicia/para gravação
            if k in (ord('r'), ord('R')):
                recording = not recording
                if recording:
                    buffer_seq = []
                    last_info_time = time.time()
                else:
                    # salvar sequência se tiver conteúdo
                    if len(buffer_seq) > 0:
                        arr = np.stack(buffer_seq, axis=0).astype(np.float32)  # (T, 63)
                        out_dir = DATA_DIR / CLASSES[selected_idx]
                        out_dir.mkdir(exist_ok=True, parents=True)
                        out_path = out_dir / f"seq_{int(time.time())}.npy"
                        np.save(out_path.as_posix(), arr)
                        saved_count[CLASSES[selected_idx]] += 1
                        print(f" Salvo: {out_path}  shape={arr.shape}")
                    else:
                        print("Sequência vazia, nada salvo.")

            # Sair
            if k in (27, ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
