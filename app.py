import streamlit as st
import numpy as np
import librosa
import pickle
import io

from tensorflow.keras.models import load_model
from streamlit_mic_recorder import mic_recorder


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.3);
        margin-top: 15px;
    }

    .result-emotion {
        font-size: 36px;
        font-weight: 700;
        margin: 10px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    model = load_model("emotion_model.keras")

    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    return model, encoder


model, encoder = load_emotion_model()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_source):

    # Load audio
    audio, sample_rate = librosa.load(
        audio_source,
        sr=None,
        mono=True
    )

    # -----------------------------------------------------
    # Make audio exactly 3 seconds
    # -----------------------------------------------------

    target_length = sample_rate * 3

    if len(audio) < target_length:

        # Add silence if audio is shorter than 3 seconds
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        # Use first 3 seconds if audio is longer
        audio = audio[:target_length]

    # -----------------------------------------------------
    # Extract MFCC
    # -----------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Take mean of MFCC features
    mfcc_mean = np.mean(
        mfcc.T,
        axis=0
    )

    # -----------------------------------------------------
    # Reshape for CNN
    # -----------------------------------------------------

    features = mfcc_mean.reshape(
        1,
        40,
        1
    )

    return features


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🎤 Speech Emotion Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered speech emotion detection using MFCC & CNN'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PROJECT INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🧠 Model",
        "CNN"
    )

with col2:

    st.metric(
        "🎵 Dataset",
        "RAVDESS"
    )

with col3:

    st.metric(
        "😊 Emotions",
        "8"
    )


st.divider()


# =========================================================
# VOICE INPUT
# =========================================================

st.markdown(
    '<div class="section-title">🎧 Voice Input</div>',
    unsafe_allow_html=True
)

st.write(
    "Choose whether you want to upload an audio file "
    "or record your voice directly."
)


input_method = st.radio(
    "Choose input method:",
    [
        "📁 Upload WAV",
        "🎙️ Record Voice"
    ],
    horizontal=True
)


# These variables are always created
# This prevents NameError problems.

audio_file = None
audio_bytes = None


# =========================================================
# OPTION 1 — UPLOAD WAV
# =========================================================

if input_method == "📁 Upload WAV":

    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        st.success(
            "Audio uploaded successfully! ✅"
        )

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        audio_file = uploaded_file


# =========================================================
# OPTION 2 — RECORD VOICE
# =========================================================

else:

    st.write(
        "Click Start Recording, speak clearly, "
        "then click Stop Recording."
    )

    recorded_audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        format="wav",
        key="voice_recorder"
    )

    if recorded_audio is not None:

        st.success(
            "Voice recorded successfully! ✅"
        )

        audio_bytes = recorded_audio["bytes"]

        st.audio(
            audio_bytes,
            format="audio/wav"
        )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

analyze_button = st.button(
    "🔮 Analyze Emotion",
    use_container_width=True
)


# =========================================================
# ANALYZE AUDIO
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Check whether audio exists
    # -----------------------------------------------------

    if audio_file is None and audio_bytes is None:

        st.warning(
            "⚠️ Please upload or record an audio file first."
        )

    else:

        with st.spinner(
            "🧠 AI is analyzing the speech..."
        ):

            try:

                # =================================================
                # UPLOADED AUDIO
                # =================================================

                if audio_file is not None:

                    features = extract_features(
                        audio_file
                    )


                # =================================================
                # RECORDED AUDIO
                # =================================================

                else:

                    # Convert recorded bytes into
                    # an in-memory file

                    audio_stream = io.BytesIO(
                        audio_bytes
                    )

                    features = extract_features(
                        audio_stream
                    )


                # =================================================
                # MODEL PREDICTION
                # =================================================

                prediction = model.predict(
                    features,
                    verbose=0
                )


                # Get highest probability
                predicted_index = np.argmax(
                    prediction
                )


                # Convert index to emotion name
                predicted_emotion = (
                    encoder.inverse_transform(
                        [predicted_index]
                    )[0]
                )


                # Calculate confidence
                confidence = (
                    prediction[0][predicted_index]
                    * 100
                )


                # =================================================
                # RESULT CARD
                # =================================================

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.write(
                    "🎯 Detected Emotion"
                )

                st.markdown(
                    f'<div class="result-emotion">'
                    f'{predicted_emotion.upper()}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.2f}%"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # =================================================
                # EMOTION PROBABILITIES
                # =================================================

                st.divider()

                st.subheader(
                    "📊 Emotion Probabilities"
                )

                emotions = encoder.classes_

                probability_columns = st.columns(4)


                for i, emotion in enumerate(emotions):

                    probability = (
                        prediction[0][i] * 100
                    )

                    with probability_columns[i % 4]:

                        st.write(
                            f"**{emotion.capitalize()}**"
                        )

                        st.progress(
                            float(
                                prediction[0][i]
                            )
                        )

                        st.caption(
                            f"{probability:.2f}%"
                        )


            except Exception as e:

                st.error(
                    f"❌ Unable to analyze audio: {e}"
                )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

st.subheader(
    "ℹ️ About This Project"
)

about_col1, about_col2 = st.columns(2)


# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------

with about_col1:

    st.markdown(
        """
        <div class="info-card">

        **How it works**

        🎤 Voice Input  
        ↓  
        📊 MFCC Feature Extraction  
        ↓  
        🧠 CNN Deep Learning Model  
        ↓  
        🔮 Emotion Prediction

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# SUPPORTED EMOTIONS
# ---------------------------------------------------------

with about_col2:

    st.markdown(
        """
        <div class="info-card">

        **Supported Emotions**

        😐 Neutral  
        😌 Calm  
        😊 Happy  
        😢 Sad  
        😡 Angry  
        😨 Fearful  
        🤢 Disgust  
        😲 Surprised

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Speech Emotion Recognition • "
    "RAVDESS + MFCC + CNN"
)