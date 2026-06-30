import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
from torchvision import transforms, models

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLING  (this block only changes how things LOOK, not the logic)
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root{
  --bg:#0b0e1a;
  --card:#151a2e;
  --card-2:#1b2138;
  --ink:#e7e9f5;
  --muted:#9499bd;
  --primary:#818cf8;
  --primary-2:#a78bfa;
  --accent:#22d3ee;
  --good:#34d399;
  --warn:#fbbf24;
  --border:rgba(255,255,255,.08);
  --shadow:0 16px 44px -18px rgba(0,0,0,.7);
}

/* hide default streamlit chrome for a cleaner report look */
#MainMenu, footer, header {visibility:hidden;}

.stApp{
  background:
    radial-gradient(1100px 520px at 100% -10%, rgba(129,140,248,.18) 0%, transparent 55%),
    radial-gradient(900px 520px at -10% 0%, rgba(34,211,238,.12) 0%, transparent 50%),
    var(--bg);
  color:var(--ink);
}

html, body, [class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
  color:var(--ink);
}

.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1180px;}

/* generic streamlit text on dark */
.stApp p, .stApp li, .stApp span, .stApp label{color:var(--ink);}
.stCaption, [data-testid="stCaptionContainer"]{color:var(--muted) !important;}

/* HERO */
.hero{
  background:linear-gradient(120deg,#6366f1 0%,#8b5cf6 55%,#a855f7 100%);
  border-radius:24px;
  padding:38px 40px;
  color:#fff;
  box-shadow:0 24px 55px -22px rgba(99,102,241,.75);
  position:relative;
  overflow:hidden;
  margin-bottom:26px;
}
.hero:after{
  content:""; position:absolute; right:-60px; top:-60px;
  width:240px; height:240px; border-radius:50%;
  background:rgba(255,255,255,.14);
}
.hero:before{
  content:""; position:absolute; right:90px; bottom:-90px;
  width:200px; height:200px; border-radius:50%;
  background:rgba(255,255,255,.08);
}
.hero .tag{
  display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; padding:6px 13px; border-radius:999px;
  background:rgba(255,255,255,.2); backdrop-filter:blur(4px); margin-bottom:14px; color:#fff;
}
.hero h1{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:2.3rem; line-height:1.1; margin:0 0 8px 0; color:#fff;
}
.hero p{font-size:1rem; opacity:.95; margin:0; max-width:640px; color:#fff;}

/* CARDS / PANELS */
.panel{
  background:linear-gradient(180deg,var(--card-2),var(--card));
  border:1px solid var(--border);
  border-radius:20px; padding:24px; box-shadow:var(--shadow);
}
.panel-title{
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:700;
  font-size:1.05rem; margin-bottom:18px; display:flex; align-items:center; gap:8px;
  color:var(--ink);
}

.stat-grid{display:grid; grid-template-columns:1fr 1fr; gap:14px;}
.stat-card{
  background:linear-gradient(180deg,#1d2340,#171c31);
  border:1px solid var(--border); border-radius:16px; padding:16px 18px;
}
.stat-card .ico{font-size:1.4rem; line-height:1;}
.stat-card .lab{font-size:.74rem; font-weight:600; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; margin:8px 0 4px;}
.stat-card .val{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:1.45rem; color:var(--ink);}
.stat-card.wide{grid-column:1 / -1;}

/* confidence bar */
.conf-block{margin-top:18px;}
.conf-row{display:flex; justify-content:space-between; align-items:baseline;
  font-size:.84rem; font-weight:600; color:var(--muted); margin-bottom:7px;}
.conf-row .pct{font-family:'Plus Jakarta Sans',sans-serif; font-size:1.05rem;
  font-weight:800; color:var(--accent);}
.bar-wrap{height:11px; background:rgba(255,255,255,.07); border-radius:999px; overflow:hidden;}
.bar-fill{height:100%; border-radius:999px;
  background:linear-gradient(90deg,var(--primary),var(--accent));}
.meta-row{margin-top:16px; font-size:.78rem; color:var(--muted);
  border-top:1px dashed var(--border); padding-top:12px;}

/* step cards */
.step{
  background:linear-gradient(180deg,var(--card-2),var(--card));
  border:1px solid var(--border); border-radius:18px;
  padding:18px 20px; box-shadow:var(--shadow); height:100%;
}
.step .num{
  width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; font-weight:800; color:#fff; font-size:.9rem;
  background:linear-gradient(135deg,var(--primary),var(--primary-2)); margin-bottom:12px;
}
.step h4{margin:0 0 12px; font-family:'Plus Jakarta Sans',sans-serif; font-size:.98rem; color:var(--ink);}
.step .kv{display:flex; justify-content:space-between; font-size:.86rem;
  padding:6px 0; border-bottom:1px dashed var(--border);}
.step .kv:last-child{border-bottom:none;}
.step .kv b{color:var(--ink);} .step .kv span{color:var(--muted);}

/* verdict banner */
.verdict{border-radius:18px; padding:20px 22px; margin-top:6px;
  font-size:.95rem; line-height:1.5; box-shadow:var(--shadow);}
.verdict.ok{background:linear-gradient(120deg,rgba(16,185,129,.16),rgba(16,185,129,.07));
  border:1px solid rgba(52,211,153,.35); color:#6ee7b7;}
.verdict.neutral{background:linear-gradient(120deg,rgba(59,130,246,.16),rgba(59,130,246,.07));
  border:1px solid rgba(96,165,250,.35); color:#93c5fd;}
.verdict b{font-weight:800; color:#fff;}
.verdict i{opacity:.85;}
.verdict .vt{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:1rem; display:block; margin-bottom:4px; color:inherit;}

/* pill */
.pill{display:inline-block; padding:4px 11px; border-radius:999px;
  font-size:.76rem; font-weight:700;}
.pill.green{background:rgba(16,185,129,.18); color:#6ee7b7;}
.pill.amber{background:rgba(245,158,11,.2); color:#fcd34d;}
.pill.blue{background:rgba(59,130,246,.2); color:#93c5fd;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:8px; background:transparent;}
.stTabs [data-baseweb="tab"]{
  background:var(--card); border:1px solid var(--border); border-radius:13px;
  padding:10px 18px; font-weight:600; color:var(--muted); box-shadow:var(--shadow);
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--primary),var(--primary-2)) !important;
  color:#fff !important; border-color:transparent !important;
}

/* uploader */
[data-testid="stFileUploader"]{
  background:var(--card); border:2px dashed rgba(255,255,255,.16); border-radius:18px; padding:12px;
}
[data-testid="stFileUploader"] *{color:var(--ink) !important;}

/* expander */
[data-testid="stExpander"]{border:1px solid var(--border); border-radius:14px;
  background:var(--card); overflow:hidden;}
[data-testid="stExpander"] summary{color:var(--ink);}

/* sidebar */
[data-testid="stSidebar"]{background:#0d1120; border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--ink);}
[data-testid="stSidebar"] .sb-title{font-family:'Plus Jakarta Sans',sans-serif;
  font-weight:800; font-size:1.1rem; margin-bottom:4px; color:var(--ink);}
[data-testid="stSidebar"] .sb-sub{font-size:.8rem; color:var(--muted) !important; margin-bottom:18px;}
.sb-row{display:flex; justify-content:space-between; font-size:.82rem;
  padding:9px 0; border-bottom:1px solid var(--border);}
.sb-row span{color:var(--muted) !important;} .sb-row b{color:var(--ink);}
.sb-legend{font-size:.82rem; color:var(--muted) !important; line-height:1.6;}

.section-h{font-family:'Plus Jakarta Sans',sans-serif; font-weight:800;
  font-size:1.25rem; margin:8px 0 4px; color:var(--ink);}
.section-sub{color:var(--muted); font-size:.9rem; margin-bottom:14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── small HTML helpers ──────────────────────────────────────────────────────
def stat_card(icon, label, value, wide=False):
    cls = "stat-card wide" if wide else "stat-card"
    return (f'<div class="{cls}"><div class="ico">{icon}</div>'
            f'<div class="lab">{label}</div><div class="val">{value}</div></div>')


def confidence_block(title, pct):
    return (f'<div class="conf-block"><div class="conf-row"><span>{title}</span>'
            f'<span class="pct">{pct:.1f}%</span></div>'
            f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct:.1f}%">'
            f'</div></div></div>')


# ════════════════════════════════════════════════════════════════════════════
# LOAD MODELS  (unchanged logic)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_age_gender_model():
    return tf.keras.models.load_model("best_model2.keras")


@st.cache_resource
def load_hair_model():
    device = torch.device('cpu')
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)          # 0=short, 1=long
    m.load_state_dict(torch.load("hair_model_utk.pth", map_location=device))
    m.eval()
    return m


age_gender_model = load_age_gender_model()
hair_model = load_hair_model()

hair_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">🧠 Vision Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Age & Gender Recognition · Internship Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-row"><span>Age / Gender</span><b>CNN</b></div>
    <div class="sb-row"><span>Hair length</span><b>ResNet18</b></div>
    <div class="sb-row"><span>Dataset</span><b>UTKFace</b></div>
    <div class="sb-row"><span>Input</span><b>128 × 128</b></div>
    <div class="sb-row"><span>Framework</span><b>TF + PyTorch</b></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title" style="font-size:.95rem;">Legend</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-legend">
    <span class="pill amber">Hair rule</span> applied for ages 20–30<br><br>
    <span class="pill blue">Facial</span> used outside that range
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <span class="tag">AI · Computer Vision</span>
  <h1>Age &amp; Gender Recognition</h1>
  <p>A deep-learning facial-analysis demo built on the UTKFace dataset, combining a
  convolutional age/gender model with a ResNet18 hair-length classifier.</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🧠  Age & Gender Detection", "💇  Task 1 — Hair-Based Gender"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-h">Face Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a face photo to estimate age and gender.</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Face Image", type=["jpg", "jpeg", "png"], key="tab1_uploader"
    )

    if uploaded_file:
        col1, col2 = st.columns([2, 1], gap="large")
        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing face..."):
            start_time = time.time()
            img = image.convert("L").resize((128, 128))
            img = np.array(img) / 255.0
            img = img.reshape(1, 128, 128, 1)
            gender_pred, age_pred = age_gender_model.predict(img, verbose=0)
            end_time = time.time()

        gender_score = float(gender_pred[0][0])
        gender = "Female" if gender_score > 0.5 else "Male"
        confidence = gender_score * 100 if gender_score > 0.5 else (1 - gender_score) * 100
        age = int(age_pred[0][0])
        g_emoji = "👩" if gender == "Female" else "👨"

        with col2:
            st.markdown(
                '<div class="panel"><div class="panel-title">📊 Prediction Results</div>'
                '<div class="stat-grid">'
                + stat_card(g_emoji, "Gender", gender)
                + stat_card("🎂", "Age", f"{age} yrs")
                + '</div>'
                + confidence_block("Gender confidence", confidence)
                + f'<div class="meta-row">⏱️ Inference time · {(end_time - start_time):.3f}s'
                  f'&nbsp;&nbsp;•&nbsp;&nbsp;Grayscale 128×128 CNN</div>'
                + '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
          <div class="panel-title">ℹ️ Model Information</div>
          <div class="stat-grid">
            <div class="stat-card"><div class="lab">Input shape</div>
              <div class="val" style="font-size:1.05rem;">128 × 128 × 1</div></div>
            <div class="stat-card"><div class="lab">Architecture</div>
              <div class="val" style="font-size:1.05rem;">CNN</div></div>
            <div class="stat-card"><div class="lab">Dataset</div>
              <div class="val" style="font-size:1.05rem;">UTKFace</div></div>
            <div class="stat-card"><div class="lab">Outputs</div>
              <div class="val" style="font-size:1.05rem;">Age + Gender</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-h">Hair-Based Gender Rule</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A rule-based experiment combining the age model '
                'with a hair-length classifier.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
      <div class="panel-title">🧩 How this works</div>
      <div class="step kv" style="box-shadow:none; border:none; padding:0;">
        <div style="font-size:.9rem; line-height:1.7; color:var(--muted);">
        <b style="color:var(--ink);">Age 20–30</b> → gender predicted from
        <b style="color:var(--ink);">hair length</b>
        (long = Female, short = Male, regardless of biological gender).<br>
        <b style="color:var(--ink);">Outside 20–30</b> → gender predicted normally
        from <b style="color:var(--ink);">facial features</b>.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Age override ───────────────────────────────────────────────────────
    with st.expander("⚙️  Demo Settings — Age Override", expanded=False):
        st.caption("The age model may not be accurate on every photo type. "
                   "Use this to manually set the age for demonstration purposes.")
        age_override = st.checkbox("Enable manual age override")
        manual_age = None
        if age_override:
            manual_age = st.slider("Set age manually", 1, 80, 25,
                                   help="Age used for the hair-rule decision")
            if 20 <= manual_age <= 30:
                st.success(f"Age {manual_age} → hair rule **will** be applied ✅")
            else:
                st.info(f"Age {manual_age} → hair rule will **not** be applied")

    st.caption("💡 For best results: clear color photo, full hair visible, good lighting, "
               "face forward.")

    uploaded_file2 = st.file_uploader(
        "Upload Face Image", type=["jpg", "jpeg", "png"], key="tab2_uploader"
    )

    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        col1, col2 = st.columns([2, 1], gap="large")

        with col1:
            st.image(image2, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            start_time = time.time()

            # Step 1: age + biological gender
            img_gray = image2.convert("L").resize((128, 128))
            img_gray = np.array(img_gray) / 255.0
            img_gray = img_gray.reshape(1, 128, 128, 1)
            gender_pred, age_pred = age_gender_model.predict(img_gray, verbose=0)

            model_age = int(age_pred[0][0])
            gender_score = float(gender_pred[0][0])
            bio_gender = "Female" if gender_score > 0.5 else "Male"

            age = manual_age if (age_override and manual_age is not None) else model_age
            age_source = "Manual override" if (age_override and manual_age is not None) \
                else "Model predicted"

            # Step 2: hair length
            img_rgb = image2.convert("RGB")
            img_tensor = hair_transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                hair_out = hair_model(img_tensor)
                hair_prob = torch.softmax(hair_out, dim=1)[0]
                hair_pred = hair_out.argmax(1).item()   # 0=short, 1=long

            hair_label = "Long" if hair_pred == 1 else "Short"
            hair_confidence = hair_prob[hair_pred].item() * 100

            # Step 3: apply rule
            in_age_range = 20 <= age <= 30
            if in_age_range:
                final_gender = "Female" if hair_pred == 1 else "Male"
                rule_used = f"Age {age} in 20–30 → hair-length rule applied"
                rule_pill = '<span class="pill amber">Hair rule</span>'
            else:
                final_gender = bio_gender
                rule_used = f"Age {age} outside 20–30 → biological gender used"
                rule_pill = '<span class="pill blue">Facial</span>'

            end_time = time.time()

        g_emoji = "👩" if final_gender == "Female" else "👨"
        hair_icon = "🟤" if hair_label == "Long" else "✂️"

        with col2:
            age_val = (f"{age} yrs <span style='font-size:.7rem;color:var(--muted);'>"
                       f"(manual)</span>") if (age_override and manual_age is not None) \
                else f"{age} yrs"
            st.markdown(
                '<div class="panel"><div class="panel-title">📊 Results</div>'
                '<div class="stat-grid">'
                + stat_card(g_emoji, "Predicted gender", final_gender, wide=True)
                + stat_card("🎂", "Age used", age_val)
                + stat_card(hair_icon, "Hair length", hair_label)
                + '</div>'
                + confidence_block("Hair confidence", hair_confidence)
                + f'<div class="meta-row">⏱️ Inference time · '
                  f'{(end_time - start_time):.3f}s</div>'
                + '</div>',
                unsafe_allow_html=True
            )

        # ── Decision explanation ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-h">Decision Breakdown</div>', unsafe_allow_html=True)

        c3, c4, c5 = st.columns(3, gap="medium")
        age_status = ('<span class="pill green">In range</span>' if in_age_range
                      else '<span class="pill blue">Out of range</span>')

        with c3:
            st.markdown(f"""
            <div class="step"><div class="num">1</div><h4>Age Check</h4>
              <div class="kv"><span>Model predicted</span><b>{model_age}</b></div>
              <div class="kv"><span>Age used</span><b>{age} ({age_source})</b></div>
              <div class="kv"><span>Status</span><b>{age_status}</b></div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="step"><div class="num">2</div><h4>Hair Detection</h4>
              <div class="kv"><span>Hair length</span><b>{hair_label}</b></div>
              <div class="kv"><span>Confidence</span><b>{hair_confidence:.1f}%</b></div>
              <div class="kv"><span>Biological gender</span><b>{bio_gender}</b></div>
            </div>""", unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="step"><div class="num">3</div><h4>Rule Applied</h4>
              <div class="kv"><span>Path</span><b>{rule_pill}</b></div>
              <div class="kv"><span>Reason</span><b style="font-size:.78rem;">{rule_used}</b></div>
              <div class="kv"><span>Final</span><b>{final_gender}</b></div>
            </div>""", unsafe_allow_html=True)

        # ── Final verdict ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if in_age_range:
            st.markdown(f"""
            <div class="verdict ok"><span class="vt">✅ Hair rule applied</span>
            Age <b>{age}</b> falls in the 20–30 range. Hair detected as
            <b>{hair_label}</b> → predicted <b>{final_gender}</b>
            <i>(biological gender “{bio_gender}” overridden by hair length).</i></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict neutral"><span class="vt">ℹ️ Standard prediction</span>
            Age <b>{age}</b> is outside the 20–30 range, so facial cues were used →
            predicted <b>{final_gender}</b> <i>(hair length ignored).</i></div>
            """, unsafe_allow_html=True)

    # ── Model info ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <div class="panel-title">ℹ️ Model Information</div>
      <div class="step kv" style="box-shadow:none;border:none;padding:0;">
        <div style="font-size:.86rem;line-height:1.8;color:var(--muted);">
        <b style="color:var(--ink);">Age model</b> — best_model.keras
        (CNN · UTKFace · grayscale 128×128)<br>
        <b style="color:var(--ink);">Hair model</b> — hair_model_utk.pth
        (ResNet18 · UTKFace · RGB 128×128)<br>
        <b style="color:var(--ink);">Training</b> — 405 manually labeled UTKFace
        images (ages 20–30)<br>
        <b style="color:var(--ink);">Rule</b> — long hair = Female, short hair = Male
        for ages 20–30
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)