# 🎤 Speech Emotion Recognition

An AI-based Speech Emotion Recognition system that detects human emotions from speech audio using **MFCC feature extraction** and a **Convolutional Neural Network (CNN)**.

## 📌 Project Overview

This project is developed as part of the **CodeAlpha Virtual Machine Learning Internship**.

The system takes speech audio as input and predicts the emotion expressed in the voice.

## 😊 Supported Emotions

The model can recognize 8 emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Streamlit Mic Recorder

## 🎵 Dataset

The model is trained using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The dataset contains emotional speech recordings from different actors.

## 🔄 ML PIPELINE

```text
Speech Audio
     ↓
Audio Preprocessing
     ↓
MFCC Feature Extraction
     ↓
CNN Model
     ↓
Emotion Prediction
     ↓
Confidence Score



🖥️ Application Features 
*Upload WAV audio files
*Record voice directly using microphone
*MFCC-based audio feature extraction
*CNN-based emotion classification
*Emotion prediction
*Confidence score
*Probability distribution for all emotions
*Streamlit web interface

📂 Project Structure
CodeAlpha_EmotionRecognition/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── extract_features.py
├── predict.py
├── emotion_recognition.py
├── main.py
├── emotion_model.keras
├── label_encoder.pkl
├── X.npy
├── y.npy
└── README.md


▶️ How to Run

Install the required libraries:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py


🎯 Internship Task
CodeAlpha Virtual Machine Learning Internship
Task: Speech Emotion Recognition

👨‍💻 Author
Atharva Kaware

