import streamlit as st
import cv2
import numpy as np
import json
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="TrafficLens — Car Colour & People Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* App shell */
.app-shell {
    min-height: 100vh;
    background: #0d0f14;
    color: #e8eaf0;
    padding: 0;
}

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 40px;
    background: #13161e;
    border-bottom: 1px solid #1e2330;
}
.topbar-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.topbar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f0f2f8;
    letter-spacing: -0.3px;
}
.topbar-badge {
    margin-left: auto;
    background: #1e2330;
    border: 1px solid #2a3040;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #6b7280;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Hero strip */
.hero {
    background: linear-gradient(135deg, #0f1520 0%, #131825 60%, #0d1018 100%);
    border-bottom: 1px solid #1a1f2e;
    padding: 48px 40px 40px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: #f0f2f8;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0;
}
.hero h1 span {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 15px;
    color: #6b7280;
    font-weight: 400;
    margin: 4px 0 0 0;
    max-width: 540px;
    line-height: 1.6;
}
.legend-row {
    display: flex;
    gap: 24px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #9ca3af;
}
.legend-dot {
    width: 12px; height: 12px;
    border-radius: 3px;
    border: 2px solid;
    flex-shrink: 0;
}

/* Main content area */
.main-content {
    padding: 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 28px;
}

/* Upload zone */
.upload-zone {
    background: #13161e;
    border: 1.5px dashed #2a3040;
    border-radius: 16px;
    padding: 0;
    transition: border-color 0.2s;
}
.upload-zone:hover {
    border-color: #3b82f6;
}

/* Control panel */
.control-panel {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 16px;
    padding: 24px;
}
.control-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #4b5563;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
.metric-card {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card.total::before { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.metric-card.blue::before  { background: #ef4444; }
.metric-card.other::before { background: #3b82f6; }
.metric-card.people::before { background: #22c55e; }
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 40px;
    font-weight: 700;
    color: #f0f2f8;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #374151;
    margin-top: 6px;
}

/* Image panels */
.panel-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 12px;
    padding: 0 4px;
}
.image-panel {
    background: #0d0f14;
    border: 1px solid #1e2330;
    border-radius: 14px;
    overflow: hidden;
}

/* Slider tweaks */
.stSlider > div > div > div > div {
    background: #3b82f6 !important;
}
.stSlider [data-testid="stSliderThumb"] {
    background: #3b82f6 !important;
    border: 2px solid #1d4ed8 !important;
}

/* Upload widget */
.stFileUploader {
    background: transparent !important;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #0d0f14 !important;
    border: 1.5px dashed #2a3040 !important;
    border-radius: 14px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* Info banner */
.info-banner {
    background: #0f1825;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    color: #60a5fa;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
    return net

@st.cache_resource
def load_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

@st.cache_resource
def load_color_model():
    model = tf.keras.models.load_model("car_color_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
           "cat","chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
CAR_CLASS_ID    = CLASSES.index("car")
BUS_CLASS_ID    = CLASSES.index("bus")
PERSON_CLASS_ID = CLASSES.index("person")

net = load_model()
hog = load_hog()
color_model, color_class_names = load_color_model()

# ── Helpers ──────────────────────────────────────────────────────────────────
def boxes_overlap(a, b, iou_thresh=0.3):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    iw = max(0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0, min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    union = max(1,(ax2-ax1)*(ay2-ay1)) + max(1,(bx2-bx1)*(by2-by1)) - inter
    return (inter/union) > iou_thresh

def contains_overlap(pb, cb, thresh=0.5):
    px1,py1,px2,py2 = pb; cx1,cy1,cx2,cy2 = cb
    iw = max(0, min(px2,cx2)-max(px1,cx1))
    ih = max(0, min(py2,cy2)-max(py1,cy1))
    return (iw*ih / max(1,(px2-px1)*(py2-py1))) > thresh

def run_ssd(img):
    h,w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)), 0.007843, (300,300), (127.5,)*3, swapRB=False)
    net.setInput(blob)
    det = net.forward()
    out = []
    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        cls  = int(det[0,0,i,1])
        box  = det[0,0,i,3:7] * np.array([w,h,w,h])
        out.append((cls, conf, *box))
    return out

def nms_filter(boxes, scores, thresh, nms_t=0.4):
    if not boxes: return []
    bbs = [[int(x1),int(y1),int(x2-x1),int(y2-y1)] for x1,y1,x2,y2 in boxes]
    idx = cv2.dnn.NMSBoxes(bbs, scores, thresh, nms_t)
    return [] if len(idx)==0 else [int(i) for i in np.array(idx).flatten()]

def colour_label(crop):
    if crop.size == 0: return "other"
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(cv2.resize(rgb,(128,128)), 0).astype("float32")
    pred = color_model.predict(inp, verbose=0)
    return "blue" if color_class_names[np.argmax(pred)].lower()=="blue" else "other"

def detect_and_annotate(image_bgr, car_conf, person_conf, person_sens):
    h,w = image_bgr.shape[:2]
    dets = run_ssd(image_bgr)

    car_boxes=[]; car_scores=[]; per_boxes=[]; per_scores=[]
    for cls,conf,x1,y1,x2,y2 in dets:
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
        if x2<=x1 or y2<=y1: continue
        if cls in (CAR_CLASS_ID,BUS_CLASS_ID) and conf>=car_conf:
            car_boxes.append((x1,y1,x2,y2)); car_scores.append(conf)
        elif cls==PERSON_CLASS_ID and conf>=person_conf:
            per_boxes.append((x1,y1,x2,y2)); per_scores.append(conf)

    blue_c=0; other_c=0; final_cars=[]
    for i in nms_filter(car_boxes, car_scores, car_conf):
        x1,y1,x2,y2 = [int(v) for v in car_boxes[i]]
        conf = car_scores[i]; final_cars.append((x1,y1,x2,y2))
        lbl = colour_label(image_bgr[y1:y2,x1:x2])
        if lbl=="blue":
            col=(0,0,255); blue_c+=1; txt=f"Blue car {conf:.2f}"
        else:
            col=(255,0,0); other_c+=1; txt=f"Car {conf:.2f}"
        # Thicker, nicer boxes
        cv2.rectangle(image_bgr,(x1,y1),(x2,y2),col,2)
        # Label background pill
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image_bgr,(x1,max(y1-24,0)),(x1+tw+8,max(y1,24)),col,-1)
        cv2.putText(image_bgr,txt,(x1+4,max(y1-6,18)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

    final_ppl=[]
    for i in nms_filter(per_boxes, per_scores, person_conf):
        x1,y1,x2,y2=[int(v) for v in per_boxes[i]]
        final_ppl.append(((x1,y1,x2,y2),per_scores[i]))

    hog_r,hog_w = hog.detectMultiScale(image_bgr,winStride=(4,4),padding=(8,8),scale=1.03)
    for (x,y,bw,bh),wt in zip(hog_r,hog_w):
        if wt<person_sens: continue
        box=(x,y,x+bw,y+bh)
        if any(contains_overlap(box,cb) for cb in final_cars): continue
        if any(boxes_overlap(box,pb[0]) for pb in final_ppl): continue
        final_ppl.append((box,float(wt)))

    for (x1,y1,x2,y2),sc in final_ppl:
        col=(0,200,80)
        cv2.rectangle(image_bgr,(x1,y1),(x2,y2),col,2)
        txt=f"Person {sc:.2f}"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)
        cv2.rectangle(image_bgr,(x1,max(y1-24,0)),(x1+tw+8,max(y1,24)),col,-1)
        cv2.putText(image_bgr,txt,(x1+4,max(y1-6,18)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

    return image_bgr, {"total":blue_c+other_c,"blue":blue_c,"other":other_c,"people":len(final_ppl)}


# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">🚦</div>
  <span class="topbar-title">TrafficLens</span>
  <span class="topbar-badge">ML Demo</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Detect. Classify. <span>Count.</span></h1>
  <p class="hero-sub">
    Upload any traffic image. The system detects every car, classifies its colour,
    and counts pedestrians — using a three-model ML pipeline.
  </p>
  <div class="legend-row">
    <div class="legend-item">
      <div class="legend-dot" style="border-color:#ef4444;"></div>
      Red box — blue car
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="border-color:#3b82f6;"></div>
      Blue box — other colours
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="border-color:#22c55e;"></div>
      Green box — person
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Controls row
with st.container():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<div class="control-panel-title">Detection settings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        car_conf = st.slider("Car confidence", 0.1, 0.9, 0.5, 0.05,
                             help="Minimum confidence to count a detected object as a car")
    with c2:
        person_conf = st.slider("Person confidence (SSD)", 0.1, 0.9, 0.3, 0.05,
                                help="Minimum confidence for SSD person detections")
    with c3:
        person_sens = st.slider("HOG sensitivity", 0.0, 1.0, 0.5, 0.05,
                                help="Lower = catches more people, higher = fewer false positives")
    st.markdown('</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"],
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div class="info-banner">
      ↑ &nbsp; Drop a traffic image above to get started — JPG or PNG, any size.
    </div>
    """, unsafe_allow_html=True)
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np  = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection pipeline…"):
        result_bgr, counts = detect_and_annotate(img_bgr.copy(), car_conf, person_conf, person_sens)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Metric cards
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card total">
        <div class="metric-label">Total cars</div>
        <div class="metric-value">{counts["total"]}</div>
        <div class="metric-sub">detected in image</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-label">Blue cars</div>
        <div class="metric-value">{counts["blue"]}</div>
        <div class="metric-sub">red box drawn</div>
      </div>
      <div class="metric-card other">
        <div class="metric-label">Other cars</div>
        <div class="metric-value">{counts["other"]}</div>
        <div class="metric-sub">blue box drawn</div>
      </div>
      <div class="metric-card people">
        <div class="metric-label">People</div>
        <div class="metric-value">{counts["people"]}</div>
        <div class="metric-sub">green box drawn</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Image panels
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="panel-label">Input image</div>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
    with col2:
        st.markdown('<div class="panel-label">Detection result</div>', unsafe_allow_html=True)
        st.image(result_rgb, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)