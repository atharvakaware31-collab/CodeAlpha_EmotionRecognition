import os
import librosa
import numpy as np

DATASET_PATH = "dataset/RAVDESS"

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

print("Starting MFCC feature extraction...\n")

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(os.listdir(actor_path)):

        if filename.endswith(".wav"):

            file_path = os.path.join(actor_path, filename)

            parts = filename.split("-")
            emotion_code = parts[2]

            emotion = emotion_labels.get(emotion_code)

            try:
                audio, sample_rate = librosa.load(file_path, duration=3)

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=40
                )

                mfcc_mean = np.mean(mfcc.T, axis=0)

                features.append(mfcc_mean)
                labels.append(emotion)

            except Exception as e:
                print("Error processing:", filename)
                print(e)

print("\n-----------------------------")
print("Feature extraction completed!")
print("Total samples:", len(features))
print("MFCC features per sample:", len(features[0]))
print("-----------------------------")

# Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

print("X shape:", X.shape)
print("y shape:", y.shape)
# Save extracted features and labels
np.save("X.npy", X)
np.save("y.npy", y)

print("\nFeatures saved successfully!")
print("X.npy and y.npy created.")