import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading data...")

X = np.load("X.npy")
y = np.load("y.npy")


# ==========================================
# 2. LOAD LABEL ENCODER
# ==========================================

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

y_encoded = encoder.transform(y)


# ==========================================
# 3. RESHAPE FEATURES
# ==========================================

X = X.reshape(X.shape[0], X.shape[1], 1)


# ==========================================
# 4. CREATE SAME TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = load_model("emotion_model.keras")


# ==========================================
# 6. PREDICT TEST DATA
# ==========================================

print("Predicting test data...")

predictions = model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=encoder.classes_
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test_encoded,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 9. DISPLAY CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

display.plot(
    xticks_rotation=45
)

plt.title("Speech Emotion Recognition - Confusion Matrix")

plt.tight_layout()

plt.show()