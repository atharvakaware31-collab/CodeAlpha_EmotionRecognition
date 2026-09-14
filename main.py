import os

# Path to RAVDESS dataset
DATASET_PATH = "dataset/RAVDESS"

# RAVDESS emotion mapping
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

print("Reading RAVDESS emotion labels...\n")

count = 0

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(os.listdir(actor_path)):

        if filename.endswith(".wav"):

            # Example filename:
            # 03-01-05-01-02-01-12.wav

            parts = filename.split("-")

            emotion_code = parts[2]

            emotion = emotion_labels.get(emotion_code, "unknown")

            print(filename, "->", emotion)

            count += 1

print("\n-----------------------------")
print("Total audio files:", count)
print("-----------------------------")
print("Emotion label extraction successful!")