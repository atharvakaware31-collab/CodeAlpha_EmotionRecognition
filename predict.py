import numpy as np
import librosa
import pickle

from tensorflow.keras.models import load_model


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = load_model("emotion_model.keras")

print("Model loaded successfully!")


# ==========================================
# 2. LOAD EMOTION LABEL ENCODER
# ==========================================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# ==========================================
# 3. SELECT AUDIO FILE
# ==========================================

audio_path = "dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"


# ==========================================
# 4. LOAD AUDIO
# ==========================================

audio, sample_rate = librosa.load(
    audio_path,
    duration=3
)


# ==========================================
# 5. EXTRACT MFCC FEATURES
# ==========================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

mfcc_mean = np.mean(mfcc.T, axis=0)


# ==========================================
# 6. PREPARE FEATURES FOR MODEL
# ==========================================

features = mfcc_mean.reshape(1, 40, 1)


# ==========================================
# 7. PREDICT EMOTION
# ==========================================

prediction = model.predict(
    features,
    verbose=0
)

predicted_index = np.argmax(prediction)

predicted_emotion = encoder.inverse_transform(
    [predicted_index]
)[0]

confidence = prediction[0][predicted_index] * 100


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("\n================================")
print("       EMOTION PREDICTION")
print("================================")

print("Predicted emotion:", predicted_emotion)
print("Confidence:", round(confidence, 2), "%")

print("================================")