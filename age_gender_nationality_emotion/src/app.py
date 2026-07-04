uimport streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from deepface import DeepFace

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nationality Detection System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── root & body ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: #151820 !important; border-bottom: 1px solid #252a38; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
footer { display: none !important; }
#MainMenu { display: none; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #151820 !important;
    border-right: 1px solid #252a38 !important;
}

/* ── main container ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── top header bar ── */
.top-bar {
    background: #151820;
    border-bottom: 1px solid #252a38;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0;
}
.top-bar-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #6c8fff;
    box-shadow: 0 0 10px #6c8fff88;
    display: inline-block;
}
.top-bar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: 0.02em;
}
.top-bar-sub { font-size: 12px; color: #7a8099; margin-left: 4px; }
.top-bar-badge {
    margin-left: auto;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(108,143,255,0.12);
    color: #6c8fff;
    border: 1px solid rgba(108,143,255,0.22);
}

/* ── section label ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #7a8099;
    padding: 14px 0 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #6c8fff;
    opacity: 0.7;
    display: inline-block;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    background: #1c2030 !important;
    border: 1.5px dashed #2e3448 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6c8fff !important; }
[data-testid="stFileUploader"] label { color: #e8eaf0 !important; font-size: 13px; }
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #7a8099 !important; }

/* ── image display ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid #252a38 !important;
    width: 100% !important;
    object-fit: cover;
}

/* ── toggle / checkbox ── */
[data-testid="stCheckbox"] {
    background: #1c2030;
    border: 1px solid #252a38;
    border-radius: 8px;
    padding: 10px 14px !important;
}
[data-testid="stCheckbox"] label {
    color: #7a8099 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
[data-testid="stCheckbox"] span { color: #7a8099 !important; }

/* ── button ── */
[data-testid="stButton"] > button {
    background: #4f6edb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 13px 0 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
[data-testid="stButton"] > button:hover { background: #6c8fff !important; }

/* ── spinner ── */
[data-testid="stSpinner"] { color: #6c8fff !important; }

/* ── divider ── */
hr { border-color: #252a38 !important; margin: 4px 0 !important; }

/* ── nationality badge ── */
.nat-badge {
    border-radius: 10px;
    padding: 16px 18px;
    border: 1px solid;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 4px;
}
.nat-dot { width:12px; height:12px; border-radius:50%; flex-shrink:0; }
.nat-label { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.09em; opacity:0.65; margin-bottom:4px; }
.nat-value { font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; line-height:1; }

/* ── metric cards ── */
.metric-card {
    background: #1c2030;
    border: 1px solid #252a38;
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
}
.metric-card-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7a8099;
    margin-bottom: 8px;
}
.metric-card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1;
}
.metric-card-unit {
    font-size: 13px;
    font-weight: 400;
    color: #7a8099;
}
.emotion-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
}
.colour-swatch {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 4px;
}
.swatch-circle {
    width: 26px; height: 26px;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.15);
    flex-shrink: 0;
}
.swatch-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #e8eaf0;
}

/* ── confidence bar ── */
.conf-wrap { margin-top: 12px; }
.conf-label { font-size: 10px; color: #7a8099; margin-bottom: 5px; }
.conf-track { height:4px; background:#252a38; border-radius:2px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:2px; }

/* ── empty state ── */
.empty-state {
    background: #1c2030;
    border: 1px solid #252a38;
    border-radius: 12px;
    padding: 48px 20px;
    text-align: center;
    color: #7a8099;
}
.empty-icon { font-size: 36px; margin-bottom: 12px; opacity: 0.4; }
.empty-text { font-size: 13px; line-height: 1.6; }

/* ── footer bar ── */
.footer-bar {
    background: #151820;
    border-top: 1px solid #252a38;
    padding: 10px 32px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.footer-pill {
    font-size: 10px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 4px;
    background: #1c2030;
    border: 1px solid #2e3448;
    color: #7a8099;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── streamlit metric override ── */
[data-testid="stMetric"] {
    background: #1c2030!important;
    border: 1px solid #252a38!important;
    border-radius: 10px!important;
    padding: 14px 16px!important;
}
[data-testid="stMetricLabel"] { color: #7a8099!important; font-size:10px!important; text-transform:uppercase; letter-spacing:0.1em; }
[data-testid="stMetricValue"] { color: #e8eaf0!important; font-family:'Space Grotesk',sans-serif!important; font-size:22px!important; font-weight:700!important; }
[data-testid="stMetricDelta"] { display:none!important; }

/* hide streamlit column gaps slightly */
[data-testid="column"] { padding: 0 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
IMG_SIZE = 128
race_map = {0: 'White', 1: 'Black/African', 2: 'Asian', 3: 'Indian', 4: 'Others'}

NAT_CONFIG = {
    'Indian':  {'color': '#34d399', 'bg': 'rgba(52,211,153,0.10)',  'border': 'rgba(52,211,153,0.25)',  'label': 'Indian'},
    'US':      {'color': '#60a5fa', 'bg': 'rgba(96,165,250,0.10)',  'border': 'rgba(96,165,250,0.25)',  'label': 'United States'},
    'African': {'color': '#f59e0b', 'bg': 'rgba(245,158,11,0.10)',  'border': 'rgba(245,158,11,0.25)',  'label': 'African'},
    'Other':   {'color': '#a78bfa', 'bg': 'rgba(167,139,250,0.10)', 'border': 'rgba(167,139,250,0.25)', 'label': 'Other'},
}

EMOTION_CONFIG = {
    'happy':    {'emoji': '😊', 'color': '#34d399', 'bg': 'rgba(52,211,153,0.15)'},
    'sad':      {'emoji': '😢', 'color': '#60a5fa', 'bg': 'rgba(96,165,250,0.15)'},
    'angry':    {'emoji': '😠', 'color': '#f87171', 'bg': 'rgba(248,113,113,0.15)'},
    'fear':     {'emoji': '😨', 'color': '#c084fc', 'bg': 'rgba(192,132,252,0.15)'},
    'disgust':  {'emoji': '🤢', 'color': '#4ade80', 'bg': 'rgba(74,222,128,0.15)'},
    'surprise': {'emoji': '😲', 'color': '#fb923c', 'bg': 'rgba(251,146,60,0.15)'},
    'neutral':  {'emoji': '😐', 'color': '#94a3b8', 'bg': 'rgba(148,163,184,0.15)'},
}

COLOUR_HEX = {
    'Black': '#111111', 'White': '#f0f0eb', 'Gray': '#6b7280',
    'Red': '#dc2626',   'Maroon': '#7f1d1d', 'Pink': '#f9a8d4',
    'Orange': '#ea580c','Brown': '#78350f',  'Tan': '#c2956c',
    'Beige': '#d4b896', 'Yellow': '#eab308', 'Olive': '#65a30d',
    'Green': '#16a34a', 'Teal': '#0d9488',   'Blue': '#2563eb',
    'Navy': '#1e3a5f',  'Purple': '#7c3aed',
}

SKIN_BRIGHTNESS_THRESHOLD = 65

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
COLOR_NAMES = {
    'Black':(0,0,0), 'White':(255,255,255), 'Gray':(128,128,128),
    'Red':(200,30,30), 'Maroon':(120,30,40), 'Pink':(230,130,180),
    'Orange':(230,140,30), 'Brown':(120,80,40), 'Tan':(190,150,110),
    'Beige':(220,200,160), 'Yellow':(220,200,40), 'Olive':(120,120,40),
    'Green':(40,140,40), 'Teal':(30,130,130), 'Blue':(40,60,180),
    'Navy':(30,40,90), 'Purple':(120,40,140),
}

def closest_color_name(rgb):
    px = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0].astype(int)
    best, best_d = None, 1e9
    for name, ref in COLOR_NAMES.items():
        rl = cv2.cvtColor(np.uint8([[ref]]), cv2.COLOR_RGB2LAB)[0][0].astype(int)
        d = int(((px - rl) ** 2).sum())
        if d < best_d:
            best, best_d = name, d
    return best

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(pil_image):
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return pil_image
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    m = int(0.2 * w)
    x1,y1 = max(x-m,0), max(y-m,0)
    x2,y2 = min(x+w+m,img.shape[1]), min(y+h+m,img.shape[0])
    return Image.fromarray(img[y1:y2,x1:x2])

def white_balance(img_rgb):
    img = img_rgb.astype(np.float32)
    avg = img.reshape(-1,3).mean(axis=0)
    scale = avg.mean() / (avg + 1e-6)
    return np.clip(img * scale, 0, 255).astype(np.uint8)

def preprocess_nationality_norm(pil_face):
    rgb  = white_balance(np.array(pil_face.convert('RGB')))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return (gray / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

def white_balanced_face(pil_face):
    rgb = white_balance(np.array(pil_face.convert('RGB')))
    return Image.fromarray(rgb)

def get_skin_tone(pil_face):
    img = np.array(pil_face.convert('RGB')).astype(float)
    h, w = img.shape[:2]
    patch = img[int(0.35*h):int(0.65*h), int(0.30*w):int(0.70*w)].reshape(-1,3)
    bright = patch.mean(axis=1)
    patch = patch[(bright > 30) & (bright < 240)]
    if len(patch) < 20:
        patch = img.reshape(-1,3)
    return patch.mean(axis=0)

def refine_with_skin_tone(probs, pil_face):
    idx_african, idx_indian = 1, 3
    top = int(np.argmax(probs))
    if top in (idx_african, idx_indian) and abs(probs[idx_indian] - probs[idx_african]) < 0.6:
        r, g, b = get_skin_tone(pil_face)
        brightness = (r + g + b) / 3
        return idx_african if brightness < SKIN_BRIGHTNESS_THRESHOLD else idx_indian
    return top

def map_to_nationality(race_label):
    if   race_label == 'Indian':         return 'Indian'
    elif race_label == 'Black/African':  return 'African'
    elif race_label == 'White':          return 'US'
    else:                                return 'Other'

def preprocess(pil_image):
    img = np.array(pil_image.convert('L'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict_emotion(pil_image):
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion'].lower()
    except Exception:
        return 'neutral'

def detect_dress_color(pil_image):
    img = np.array(pil_image.convert('RGB'))
    img = white_balance(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    H, W = img.shape[:2]
    top   = min(y + int(1.3*h), H-1)
    bottom = H
    left  = max(x + int(0.15*w), 0)
    right = min(x + int(0.85*w), W)
    if bottom - top < 10:
        return None
    region = img[top:bottom, left:right].reshape(-1,3)
    brightness = region.mean(axis=1)
    spread = region.max(axis=1).astype(int) - region.min(axis=1).astype(int)
    keep = ~((brightness > 200) & (spread < 25))
    keep &= brightness > 25
    ycrcb = cv2.cvtColor(region.reshape(-1,1,3), cv2.COLOR_RGB2YCrCb).reshape(-1,3)
    cr, cb = ycrcb[:,1], ycrcb[:,2]
    skin = (cr>133)&(cr<173)&(cb>77)&(cb<127)
    keep &= ~skin
    filtered = region[keep]
    if len(filtered) < 30:
        filtered = region[brightness > 25]
    if len(filtered) < 10:
        return None
    km = KMeans(n_clusters=min(3,len(filtered)), n_init=10, random_state=42).fit(filtered)
    centers = km.cluster_centers_
    counts = np.bincount(km.labels_, minlength=len(centers))
    hsv = cv2.cvtColor(centers.astype(np.uint8).reshape(-1,1,3),
                       cv2.COLOR_RGB2HSV).reshape(-1,3)
    scores = hsv[:,1].astype(float) * np.sqrt(counts)
    dominant = centers[int(np.argmax(scores))].astype(int)
    return closest_color_name(tuple(dominant))

@st.cache_resource
def get_models():
    age_gender_model = load_model('best_model.keras')
    nationality_model = load_model('age_gender_race_model.keras')
    return age_gender_model, nationality_model

def run_predict(pil_image, force_dress=False):
    age_gender_model, nationality_model = get_models()
    face = crop_face(pil_image)

    age_input = preprocess(pil_image)
    ag = age_gender_model.predict(age_input, verbose=0)
    gender = "Male" if ag[0][0][0] < 0.5 else "Female"
    age = int(ag[1][0][0])

    nat_input = preprocess_nationality_norm(face)
    wb_face   = white_balanced_face(face)
    nat = nationality_model.predict(nat_input, verbose=0)
    race_probs = nat[2][0]
    race_idx   = refine_with_skin_tone(race_probs, wb_face)
    race_label = race_map[race_idx]
    nationality = map_to_nationality(race_label)
    confidence = float(race_probs[race_idx]) * 100

    emotion = predict_emotion(pil_image)

    result = {
        'nationality': nationality,
        'race_label':  race_label,
        'confidence':  confidence,
        'emotion':     emotion,
        'gender':      gender,
    }
    if nationality == 'Indian':
        result['age']   = age
        result['dress'] = detect_dress_color(pil_image)
    elif nationality == 'US':
        result['age']   = age
    elif nationality == 'African':
        result['dress'] = detect_dress_color(pil_image)

    if force_dress and 'dress' not in result:
        result['dress'] = detect_dress_color(pil_image)

    return result

# ── RENDER HELPERS ───────────────────────────────────────────────────────────
def section_label(text, dot_color='#6c8fff'):
    st.markdown(f"""
    <div class="section-label">
      <span class="section-dot" style="background:{dot_color}"></span>
      {text}
    </div>""", unsafe_allow_html=True)

def nat_badge(nationality):
    cfg = NAT_CONFIG.get(nationality, NAT_CONFIG['Other'])
    st.markdown(f"""
    <div class="nat-badge" style="background:{cfg['bg']};border-color:{cfg['border']}">
      <div class="nat-dot" style="background:{cfg['color']}"></div>
      <div>
        <div class="nat-label" style="color:{cfg['color']}">Detected nationality</div>
        <div class="nat-value" style="color:{cfg['color']}">{cfg['label']}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def emotion_card(emotion):
    key = emotion.lower()
    cfg = EMOTION_CONFIG.get(key, EMOTION_CONFIG['neutral'])
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">Emotion</div>
      <div style="font-size:28px;line-height:1;margin-bottom:8px">{cfg['emoji']}</div>
      <span class="emotion-pill" style="background:{cfg['bg']};color:{cfg['color']}">
        {emotion.capitalize()}
      </span>
    </div>""", unsafe_allow_html=True)

def age_card(age):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">Estimated age</div>
      <div class="metric-card-value">
        {age}<span class="metric-card-unit"> yrs</span>
      </div>
      <div class="conf-wrap">
        <div class="conf-label">± 6 years (Task 1 model)</div>
      </div>
    </div>""", unsafe_allow_html=True)

def dress_card(colour):
    if colour is None:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-card-label">Dress colour</div>
          <div style="color:#7a8099;font-size:13px;margin-top:8px">Body not visible</div>
        </div>""", unsafe_allow_html=True)
        return
    hex_col = COLOUR_HEX.get(colour, '#888888')
    border = 'rgba(255,255,255,0.12)' if colour != 'White' else 'rgba(0,0,0,0.2)'
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-card-label">Dress colour</div>
      <div class="colour-swatch" style="margin-top:8px">
        <div class="swatch-circle" style="background:{hex_col};border-color:{border}"></div>
        <div class="swatch-name">{colour}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def confidence_bar(confidence, nat):
    cfg = NAT_CONFIG.get(nat, NAT_CONFIG['Other'])
    st.markdown(f"""
    <div class="metric-card" style="margin-top:2px">
      <div class="metric-card-label">Race model confidence</div>
      <div style="display:flex;align-items:center;gap:14px;margin-top:6px">
        <div style="flex:1">
          <div class="conf-track">
            <div class="conf-fill" style="width:{min(confidence,100):.0f}%;background:{cfg['color']}"></div>
          </div>
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:{cfg['color']}">
          {confidence:.0f}%
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def empty_results():
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">◎</div>
      <div class="empty-text">
        Upload a face photo and click<br>
        <strong style="color:#e8eaf0">Analyse image</strong> to see predictions
      </div>
    </div>""", unsafe_allow_html=True)

# ── TOP BAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <span class="top-bar-dot"></span>
  <span class="top-bar-title">Nationality Detection System
    <span class="top-bar-sub">Task 6</span>
  </span>
  <span class="top-bar-badge">UTKFace · DeepFace · OpenCV</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── MAIN COLUMNS ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="medium")

with left:
    section_label("Input image")

    uploaded = st.file_uploader(
        "Drop a face photo here",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        st.markdown(
            f"<div style='font-size:11px;color:#7a8099;margin-top:4px'>{uploaded.name}</div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    force_dress = st.checkbox(
        "Show dress colour for all nationalities (testing)",
        value=False,
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "Analyse image",
        disabled=(uploaded is None),
        use_container_width=True,
    )

with right:
    section_label("Results", dot_color='#34d399')

    if not uploaded:
        empty_results()

    elif predict_clicked:
        with st.spinner("Running models…"):
            try:
                result = run_predict(image, force_dress=force_dress)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.session_state['last_result'] = result

        nat = result['nationality']
        nat_badge(nat)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Row 1 — always emotion, plus age if applicable
        if 'age' in result:
            c1, c2 = st.columns(2)
            with c1: emotion_card(result['emotion'])
            with c2: age_card(result['age'])
        else:
            emotion_card(result['emotion'])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Dress colour if applicable
        if 'dress' in result:
            dress_card(result['dress'])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # For Other nationality — show the raw race label
        if nat == 'Other':
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-card-label">Ethnicity class</div>
              <div class="metric-card-value" style="font-size:18px">{result['race_label']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        confidence_bar(result['confidence'], nat)

    elif 'last_result' in st.session_state:
        # Show previous result without re-running
        result = st.session_state['last_result']
        nat = result['nationality']
        nat_badge(nat)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if 'age' in result:
            c1, c2 = st.columns(2)
            with c1: emotion_card(result['emotion'])
            with c2: age_card(result['age'])
        else:
            emotion_card(result['emotion'])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if 'dress' in result:
            dress_card(result['dress'])
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if nat == 'Other':
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-card-label">Ethnicity class</div>
              <div class="metric-card-value" style="font-size:18px">{result['race_label']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        confidence_bar(result['confidence'], nat)

    else:
        empty_results()

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  <span class="footer-pill">age_gender_race_model_2.keras</span>
  <span class="footer-pill">best_model.keras</span>
  <span class="footer-pill">DeepFace</span>
  <span class="footer-pill">OpenCV k-means</span>
  <span class="footer-pill">Haar cascade</span>
</div>
""", unsafe_allow_html=True)