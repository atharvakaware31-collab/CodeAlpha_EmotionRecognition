import os

DATASET_PATH = "dataset/RAVDESS"

print("Checking RAVDESS dataset...")

actors = os.listdir(DATASET_PATH)

print("Number of actor folders:", len(actors))
print("First few folders:", actors[:5])

# Count audio files
audio_count = 0

for actor in actors:
    actor_path = os.path.join(DATASET_PATH, actor)

    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                audio_count += 1

print("Total WAV audio files:", audio_count)
print("Dataset check completed successfully!")