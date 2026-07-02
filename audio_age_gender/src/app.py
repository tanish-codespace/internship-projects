"""
Voice Gender + Age + Emotion Detector — Streamlit GUI
=======================================
Pipeline:
  1. Predict gender. If female -> reject with "Upload male voice." and stop.
  2. If male -> predict age group (20s / 30s / 40s / 50s / senior_60plus).
  3. If senior_60plus -> flag as senior citizen AND detect emotion (angry/happy/sad).
     Otherwise -> just show the predicted age group.

Needs six files in the same folder as this script (download from your
Kaggle notebooks' Output tabs):
    gender_model.joblib, gender_scaler.joblib
    age_model.joblib, age_scaler.joblib
    emotion_model.joblib, emotion_scaler.joblib

Run with:
    pip install --upgrade streamlit librosa joblib scikit-learn numpy
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
GENDER_MODEL_PATH = "gender_model.joblib"
GENDER_SCALER_PATH = "gender_scaler.joblib"
AGE_MODEL_PATH = "age_model.joblib"
AGE_SCALER_PATH = "age_scaler.joblib"
EMOTION_MODEL_PATH = "emotion_model.joblib"
EMOTION_SCALER_PATH = "emotion_scaler.joblib"

N_MFCC = 13              # for gender + age models
N_MFCC_EMO = 40          # for emotion model (richer feature set)
SAMPLE_RATE = 16000
SENIOR_LABEL = "senior_60plus"

AGE_DISPLAY = {
    "twenties": "20s",
    "thirties": "30s",
    "fourties": "40s",
    "fifties": "50s",
}


@st.cache_resource
def load_models():
    gender_clf = joblib.load(GENDER_MODEL_PATH)
    gender_scaler = joblib.load(GENDER_SCALER_PATH)
    age_clf = joblib.load(AGE_MODEL_PATH)
    age_scaler = joblib.load(AGE_SCALER_PATH)
    emotion_clf = joblib.load(EMOTION_MODEL_PATH)
    emotion_scaler = joblib.load(EMOTION_SCALER_PATH)
    return gender_clf, gender_scaler, age_clf, age_scaler, emotion_clf, emotion_scaler


# -----------------------------------------------------------------------
# Feature extraction for GENDER + AGE models (30 features)
# -----------------------------------------------------------------------
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    f0_clean = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0_clean) if f0_clean.size > 0 else 0.0

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.concatenate([mfcc_mean, [f0_mean, spectral_centroid, spectral_bandwidth, zcr]])


# -----------------------------------------------------------------------
# Feature extraction for EMOTION model (163 features — MUST match CREMA-D training)
# -----------------------------------------------------------------------
def extract_emotion_features(file_path: str):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_EMO)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
    mel_mean = np.mean(mel, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    extras = [np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms), np.mean(sc), np.std(sc)]

    return np.concatenate([
        mfcc_mean, mfcc_std, mfcc_delta_mean,
        chroma_mean, chroma_std, mel_mean, extras
    ])


def predict_with(feats, clf, scaler):
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    pred = clf.predict(feats_scaled)[0]
    confidence = float(np.max(clf.predict_proba(feats_scaled)[0]))
    return pred, confidence


def save_temp_audio(audio_obj) -> str:
    name = getattr(audio_obj, "name", None)
    suffix = os.path.splitext(name)[1] if name else ""
    if not suffix:
        suffix = ".wav"

    data = audio_obj.getvalue() if hasattr(audio_obj, "getvalue") else audio_obj.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.set_page_config(page_title="Voice Age & Emotion Detector", page_icon="mic")
st.title("Voice Age & Emotion Detector")
st.write("Record your voice or upload a file. Male voices only — the model will reject female voices.")

try:
    gender_clf, gender_scaler, age_clf, age_scaler, emotion_clf, emotion_scaler = load_models()
except FileNotFoundError as e:
    st.error(
        f"Missing model file: {e}. Make sure all six .joblib files (gender, age, emotion "
        "models + scalers) are in this script's folder."
    )
    st.stop()

tab_record, tab_upload = st.tabs(["Record from microphone", "Upload a file"])

audio_source = None

with tab_record:
    st.write("Click the microphone button, allow browser mic access, and speak for a few seconds.")
    mic_recording = st.audio_input("Record a voice message")
    if mic_recording is not None:
        audio_source = mic_recording

with tab_upload:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_source = uploaded_file

if audio_source is not None:
    tmp_path = save_temp_audio(audio_source)
    st.audio(audio_source)

    with st.spinner("Analyzing voice..."):
        feats = extract_features(tmp_path)

        # Step 1: gender gate
        if feats is None:
            gender_pred = None
        else:
            gender_pred, gender_conf = predict_with(feats, gender_clf, gender_scaler)

        # If we reach the senior branch we'll also need emotion features
        emotion_result = None
        if feats is not None and gender_pred == "male":
            age_pred, age_conf = predict_with(feats, age_clf, age_scaler)
            if age_pred == SENIOR_LABEL:
                emo_feats = extract_emotion_features(tmp_path)
                if emo_feats is not None:
                    emotion_result = predict_with(emo_feats, emotion_clf, emotion_scaler)

    os.remove(tmp_path)

    if feats is None:
        st.error("Could not process this audio — try recording again, speaking a bit louder or longer.")
    elif gender_pred == "female":
        st.error("Upload male voice.")
    else:
        st.success(f"Gender: **MALE** (confidence: {gender_conf * 100:.1f}%)")

        if age_pred == SENIOR_LABEL:
            st.warning(f"Senior citizen detected (60+) — confidence: {age_conf * 100:.1f}%")
            if emotion_result is not None:
                emotion_pred, emotion_conf = emotion_result
                st.info(f"Detected emotion: **{emotion_pred.upper()}** (confidence: {emotion_conf * 100:.1f}%)")
            else:
                st.info("Could not detect emotion from this audio.")
        else:
            age_display = AGE_DISPLAY.get(age_pred, age_pred.upper())
            st.success(f"Age group: **{age_display}** (confidence: {age_conf * 100:.1f}%)")