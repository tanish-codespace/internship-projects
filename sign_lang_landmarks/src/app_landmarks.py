import streamlit as st, numpy as np, cv2, pickle, os
from collections import deque
from datetime import datetime
from PIL import Image
import mediapipe as mp

MODEL_PATH = "sign_model.pkl"
CONF = 0.60
START_HOUR, END_HOUR = 18, 22  # set 0 and 24 to test anytime
EMO = {"Hello":"👋","Yes":"✊","No":"👌","ILoveYou":"🤟","Water":"💧"}

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def time_open():
    h = datetime.now().hour
    return START_HOUR <= h < END_HOUR

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

def normalize(lms):
    pts = np.array(lms, dtype=np.float32)
    pts = pts - pts[0]
    m = np.max(np.linalg.norm(pts, axis=1))
    if m > 0: pts = pts / m
    return pts.flatten()

def predict(model, coords):
    x = normalize(coords).reshape(1, -1)
    probs = model.predict_proba(x)[0]
    i = int(np.argmax(probs))
    return model.classes_[i], float(probs[i]), probs, model.classes_

def nice(n): return n.replace("_", " ")

def image_tab(model):
    st.markdown("### 📸 Upload an Image")
    up = st.file_uploader("Choose image", type=["jpg","jpeg","png","webp"])
    if not up: return
    img = np.array(Image.open(up).convert("RGB"))
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as h:
        res = h.process(img)
    disp, coords = img.copy(), None
    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(disp, hl, mp_hands.HAND_CONNECTIONS)
        coords = [(lm.x, lm.y) for lm in hl.landmark]
    c1, c2 = st.columns(2)
    c1.image(disp, use_container_width=True)
    with c2:
        if coords is None:
            st.warning("No hand detected."); return
        lab, conf, probs, classes = predict(model, coords)
        if conf >= CONF:
            st.markdown(f"## {EMO.get(lab,'🤟')} {nice(lab)}")
            st.markdown(f"Confidence: **{conf*100:.1f}%**")
        else:
            st.warning(f"Low confidence ({conf*100:.1f}%)")
        st.bar_chart({nice(c): float(p) for c, p in zip(classes, probs)})

def video_tab(model):
    st.markdown("### 🎥 Live Detection")
    run = st.toggle("▶️ Start Camera", value=False)
    fbox, pbox, prob = st.empty(), st.empty(), st.empty()
    if not run:
        fbox.info("📷 Camera off — toggle to start"); return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        st.error("Could not open webcam."); return
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    hist = deque(maxlen=12); lab, conf, avg, classes = "—", 0.0, None, None; fc = 0
    try:
        while run:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                coords = [(lm.x, lm.y) for lm in hl.landmark]
                _, _, probs, classes = predict(model, coords)
                hist.append(probs); avg = np.mean(hist, axis=0)
                i = int(np.argmax(avg)); lab, conf = classes[i], float(avg[i])
            else:
                hist.clear(); lab, conf, avg = "—", 0.0, None
            col = (76,175,80) if conf >= CONF else (255,152,0)
            txt = f"{nice(lab)} ({conf*100:.0f}%)" if conf >= CONF else "show hand..."
            disp = cv2.resize(frame, (480,360))
            cv2.putText(disp, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            fbox.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_container_width=True)
            if fc % 10 == 0:
                if conf >= CONF: pbox.markdown(f"## {EMO.get(lab,'🤟')} {nice(lab)} ({conf*100:.0f}%)")
                else: pbox.info("✋ Show a clear sign...")
                if avg is not None: prob.bar_chart({nice(c): float(p) for c,p in zip(classes, avg)})
            fc += 1
    finally:
        cap.release(); hands.close()

def main():
    st.markdown("# 🤟 Sign Language Detector")
    st.caption("Hello · Yes · No · I Love You · Water")
    model = load_model()
    with st.sidebar:
        st.metric("Status", "🟢 Open" if time_open() else "🔴 Closed")
        st.info("Hours: 6 PM – 10 PM")
        st.metric("Model", "✅" if model else "❌")
    if not time_open():
        st.markdown("## 🔒 App Unavailable\nOpens daily 6:00 PM – 10:00 PM"); return
    if model is None:
        st.error("Model not found. Run train_landmarks.py first."); return
    t1, t2 = st.tabs(["📸 Upload", "🎥 Live"])
    with t1: image_tab(model)
    with t2: video_tab(model)

if __name__ == "__main__":
    main()