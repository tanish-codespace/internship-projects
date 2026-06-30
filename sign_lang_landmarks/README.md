# 🤟 Sign Language Detector — Landmark Version (works LIVE)

Recognizes: Hello, Yes, No, I Love You, Water
Operating hours: 6 PM – 10 PM

This version uses **MediaPipe hand landmarks** instead of raw images, so it
ignores background/lighting and actually works on a live webcam.

## Setup (all on your laptop — no Kaggle needed)

1. Install:
   pip install -r requirements.txt

2. Capture landmark data (~3 min):
   python capture_landmarks.py
   - Press SPACE to start each sign, hold it and move your hand around,
     it auto-advances after enough samples. Creates landmark_data.csv

3. Train (seconds, no GPU):
   python train_landmarks.py
   - Saves model/sign_landmark_model.keras + model/class_names.pkl

4. Run the app:
   streamlit run app_landmarks.py

## Testing outside 6–10 PM
In app_landmarks.py set START_HOUR=0 and END_HOUR=24, then change back before submitting.

## Why this works when the image model didn't
The model only sees your hand's joint geometry (normalized so position and
distance don't matter). Background, lighting, clothing, and skin tone never
reach the model, so live conditions match training.
