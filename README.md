# Internship Projects — AI/ML Tasks

Six computer vision and audio ML projects built during my internship. Each task has its own folder with `src/` (code + models) and `report/` (PDF/DOCX).

---

## Task 1 — Age, Gender & Hair Detection

**Problem Statement**
Predict a person's age, gender, and hair color from a facial image in real time.

**Dataset**
UTKFace dataset — 20,000+ face images labelled with age, gender, and ethnicity. Hair labels were manually curated using a custom labeller script.

**Methodology**
- Age & gender: Custom CNN trained with TensorFlow/Keras (`best_model2.keras`)
- Hair color: ResNet-based PyTorch model (`hair_model_utk.pth`) trained on UTKFace crops
- Frontend: Streamlit web app (`app.py`)

**Results**
- Age MAE ≈ 5 years on test set
- Gender accuracy ≈ 90%
- Hair classification accuracy ≈ 85%

**How to Run**
```bash
cd age_gender_detection/src
pip install -r requirements.txt
streamlit run app.py
```

---

## Task 2 — Senior Citizen Detection System

**Problem Statement**
Detect senior citizens (age 60+) via webcam in real time and automatically log each visit with a timestamp.

**Dataset**
UTKFace dataset used to train the age regression model. Live webcam feed used for inference.

**Methodology**
- Age estimation via Keras CNN (`best_model.keras`)
- OpenCV for real-time face detection from webcam
- Visits where predicted age ≥ 60 are logged to `senior_visit_log.csv`
- Streamlit dashboard (`senior_app.py`) displays live feed + visit history

**Results**
- Real-time detection at 15–20 FPS
- Age threshold (60+) classification accuracy ≈ 88%
- Visit logs exported as CSV for records

**How to Run**
```bash
cd senior_model/src
pip install streamlit tensorflow opencv-python pandas pillow
streamlit run senior_app.py
```

---

## Task 3 — Age, Gender, Nationality & Emotion Detection

**Problem Statement**
Build a multi-attribute face analysis system that simultaneously predicts age, gender, nationality/race, and emotion from an image.

**Dataset**
FairFace dataset for nationality/race labels combined with UTKFace for age and gender. DeepFace used for emotion inference.

**Methodology**
- Custom Keras model (`age_gender_race_model_2.keras`) trained on FairFace + UTKFace
- DeepFace library for emotion analysis
- KMeans colour clustering for skin tone analysis
- Streamlit frontend (`app.py`)

**Results**
- Race/nationality classification accuracy ≈ 72%
- Gender accuracy ≈ 91%
- Age MAE ≈ 6 years
- Emotion detection via DeepFace (7 classes)

**How to Run**
```bash
cd age_gender_nationality_emotion/src
pip install streamlit tensorflow opencv-python deepface pillow scikit-learn
streamlit run app.py
```

---

## Task 4 — Car Detection & Color Classification

**Problem Statement**
Detect cars in an image and classify their color automatically.

**Dataset**
VCoR (Vehicle Color Recognition) dataset for color classification training. MobileNet-SSD pretrained on COCO for car detection.

**Methodology**
- Car detection: MobileNetSSD (`MobileNetSSD_deploy.caffemodel`) via OpenCV DNN
- Color classification: Keras CNN (`car_color_model.keras`) trained on cropped car images
- 10 color classes defined in `class_names.json`
- Full pipeline in `app.py` with Streamlit interface

**Results**
- Car detection mAP ≈ 85% on test images
- Color classification accuracy ≈ 88% across 10 color classes

**How to Run**
```bash
cd "car_color_detector (2)/src/car_detector"
pip install -r requirements.txt
streamlit run app.py
```

---

## Task 5 — Voice-Based Age, Gender & Emotion Detection

**Problem Statement**
Predict a speaker's age, gender, and emotional state from an audio clip using voice features.

**Methodology**
- Gender: Random Forest classifier trained on MFCC + spectral features (`gender_model.joblib`)
- Age: Pretrained `wav2vec2` transformer model (audEERING) fine-tuned for age regression
- Emotion: `wav2vec2` model predicting arousal, dominance, and valence scores
- Pipeline: gender gate → if male senior → emotion analysis
- Streamlit app (`app2.py`) accepts uploaded `.wav`/`.mp3` files

**Dataset**
Custom audio samples + Common Voice dataset for gender training. audEERING pretrained models for age/emotion (downloaded from HuggingFace on first run).

**Results**
- Gender accuracy ≈ 87% on test clips
- Age MAE ≈ 7 years (wav2vec2)
- Emotion outputs: arousal/valence/dominance scores (0–1)

**How to Run**
```bash
cd audio_age_gender/src
pip install streamlit torch transformers==4.40.2 librosa joblib scikit-learn numpy
streamlit run app2.py
```
> wav2vec2 models download automatically from HuggingFace on first run (~1–2 GB).

---

## Task 6 — Sign Language Recognition (Landmark-Based)

**Problem Statement**
Recognise hand signs in real time using MediaPipe hand landmarks and a trained ML classifier.

**Dataset**
Custom dataset collected using `capture_landmarks.py` — hand landmark coordinates (21 keypoints × 2 = 42 features) captured via webcam for each sign class, saved to `landmark_data.csv`.

**Methodology**
- MediaPipe Hands for real-time 21-point hand landmark extraction
- MLP classifier (`MLPClassifier`, hidden layers 128→64) trained on landmark coordinates
- Model saved as `sign_model.pkl`
- `app_landmarks.py` runs live webcam inference with Streamlit

**Results**
- Test accuracy ≈ depends on signs collected; achieved >95% on 5-class subset
- Real-time inference at ~20 FPS

**How to Run**
```bash
cd sign_lang_landmarks/src

# (Optional) collect your own training data first:
python capture_landmarks.py

# Train the model:
python train_landmarks.py

# Run the app:
pip install -r requirements.txt
streamlit run app_landmarks.py
```

---

## Requirements (General)

All tasks use Python 3.9–3.11. It is recommended to use a separate virtual environment per task due to differing dependency versions.

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

Model files (`.keras`, `.joblib`, `.pkl`, `.pth`) are stored via Git LFS and will download automatically when you clone the repo.
