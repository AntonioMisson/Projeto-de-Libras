import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path("data")
CLASS_MAP = json.loads(Path("class_mapping.json").read_text(encoding="utf-8"))
INV_CLASS = {v: k for k, v in CLASS_MAP.items()}

INPUT_SIZE = 63      # 21 landmarks * (x,y,z)
SEQ_LEN = 40         # comprimento fixo (pad/truncate)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
HIDDEN = 128
LAYERS = 2
BIDIR = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T > target_len:
        return seq[:target_len]
    # pad com zeros no final
    pad_len = target_len - T
    pad = np.zeros((pad_len, seq.shape[1]), dtype=seq.dtype)
    return np.vstack([seq, pad])

def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for cls, idx in CLASS_MAP.items():
        for npy in sorted((DATA_DIR / cls).glob("*.npy")):
            arr = np.load(npy.as_posix()).astype(np.float32)  # (T, 63)
            arr = pad_or_truncate(arr, SEQ_LEN)
            X_list.append(arr)
            y_list.append(idx)
    X = np.stack(X_list, axis=0)  # (N, SEQ_LEN, 63)
    y = np.array(y_list, dtype=np.int64)
    return X, y

class SeqDataset(Dataset):
    def __init__(self, X, y, mean=None, std=None):
        self.X = X
        self.y = y
        if mean is None or std is None:
            # z-score por feature (63), agregando sobre N e T
            mean = X.reshape(-1, X.shape[-1]).mean(axis=0, keepdims=True)
            std  = X.reshape(-1, X.shape[-1]).std(axis=0, keepdims=True) + 1e-6
        self.mean = mean
        self.std = std

        self.Xn = (self.X - self.mean) / self.std

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, i):
        # retorna (seq_len, input_size), label
        return torch.from_numpy(self.Xn[i]).float(), torch.tensor(self.y[i]).long()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden, layers, num_classes, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, (hn, cn) = self.lstm(x)     # out: (B, T, H)
        last = out[:, -1, :]             # último passo temporal
        logits = self.fc(last)
        return logits

def main():
    X, y = load_dataset()
    if X.size == 0:
        raise RuntimeError("Dataset vazio. Grave sequências com record_landmarks.py")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Normalização com estatísticas do TREINO
    train_ds = SeqDataset(X_train, y_train)
    val_ds   = SeqDataset(X_val, y_val, mean=train_ds.mean, std=train_ds.std)

    # salvar mean/std
    np.savez("norm_stats.npz", mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMClassifier(INPUT_SIZE, HIDDEN, LAYERS, num_classes=len(CLASS_MAP), bidir=BIDIR).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    best_val = -1.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss, tr_ok, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_loss += loss.item() * xb.size(0)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        va_loss, va_ok, va_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_ok   += (logits.argmax(1) == yb).sum().item()
                va_total += xb.size(0)

        tr_acc = tr_ok / max(1, tr_total)
        va_acc = va_ok / max(1, va_total)
        print(f"[{epoch:03d}/{EPOCHS}] loss_tr={tr_loss/tr_total:.4f} acc_tr={tr_acc:.3f} | loss_va={va_loss/va_total:.4f} acc_va={va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "lstm_model.pt")
            print("modelo salvo: lstm_model.pt (melhor até agora)")

    print("Treino concluído.")

if __name__ == "__main__":
    main()
