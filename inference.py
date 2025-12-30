import numpy as np
import librosa
import cv2
import pickle
from collections import Counter
import tensorflow as tf

# ===== LOAD MODEL =====
model = tf.keras.models.load_model("models/cnn_lstm_emotion.h5")

with open("models/label_encoder_cnn_lstm.pkl", "rb") as f:
    le = pickle.load(f)

# ===== CONSTANTS (MUST MATCH TRAINING) =====
TARGET_SR = 16000
WINDOW_SEC = 3
HOP_SEC = 1
SEQ_LEN = 5
N_MELS = 128
IMG_SIZE = (128, 128)

# ===== AUDIO HELPERS =====

def standardize_audio(path):
    y, _ = librosa.load(path, sr=TARGET_SR)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def split_into_windows(y):
    win_len = TARGET_SR * WINDOW_SEC
    hop_len = TARGET_SR * HOP_SEC
    return [
        y[i:i + win_len]
        for i in range(0, len(y) - win_len + 1, hop_len)
    ]

def extract_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel -= mel.min()
    mel /= mel.max() + 1e-9
    return cv2.resize(mel, IMG_SIZE)

# ===== UTTERANCE-LEVEL PREDICTION =====
def predict_emotion(audio_path):
    y = standardize_audio(audio_path)

    min_len = TARGET_SR * (WINDOW_SEC + (SEQ_LEN - 1) * HOP_SEC)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    windows = split_into_windows(y)
    mels = [extract_mel(w) for w in windows]

    sequences = []
    for i in range(len(mels) - SEQ_LEN + 1):
        sequences.append(mels[i:i + SEQ_LEN])

    sequences = np.array(sequences)[..., np.newaxis]

    BATCH_SIZE = 16   # safe for CPU

    votes = Counter()
    timeline=[]

    for i in range(0, len(sequences), BATCH_SIZE):
      batch = sequences[i:i + BATCH_SIZE]
      preds = model.predict(batch, verbose=0)
      class_ids = np.argmax(preds, axis=1)

      for cid in class_ids:
        votes[cid] += 1
        timeline.append(le.inverse_transform([cid])[0])

    final_id = votes.most_common(1)[0][0]

    return {
    "final_emotion": le.inverse_transform([final_id])[0],
    "confidence": round(votes[final_id] / sum(votes.values()), 3),
    "sequence_votes": {
        le.inverse_transform([k])[0]: v for k, v in votes.items()
    },
    "total_sequences": sum(votes.values()),
    "timeline": timeline   # 👈 NEW
}

