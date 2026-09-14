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

## 🔄 ML Pipeline

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
