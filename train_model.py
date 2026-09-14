import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading extracted features...")

X = np.load("X.npy")
y = np.load("y.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ==========================================
# 2. ENCODE EMOTIONS
# ==========================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nEmotion classes:")

for number, emotion in enumerate(encoder.classes_):
    print(number, "->", emotion)

y_categorical = to_categorical(y_encoded)


# ==========================================
# 3. RESHAPE FEATURES
# ==========================================

X = X.reshape(X.shape[0], X.shape[1], 1)

print("\nNew X shape:", X.shape)
print("New y shape:", y_categorical.shape)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. BUILD CNN
# ==========================================

print("\nBuilding CNN model...")

model = Sequential([

    Conv1D(
        64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], 1)
    ),

    MaxPooling1D(pool_size=2),

    Dropout(0.25),

    Conv1D(
        128,
        kernel_size=3,
        activation="relu"
    ),

    MaxPooling1D(pool_size=2),

    Dropout(0.25),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])


# ==========================================
# 6. COMPILE
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nStarting model training... 🔥\n")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)


# ==========================================
# 9. SAVE MODEL
# ==========================================

model.save("emotion_model.keras")

print("\nModel saved as: emotion_model.keras")


# ==========================================
# 10. SAVE LABEL ENCODER
# ==========================================

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Label encoder saved as: label_encoder.pkl")

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
# ==========================================
# 11. PLOT TRAINING PERFORMANCE
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
print("================================")