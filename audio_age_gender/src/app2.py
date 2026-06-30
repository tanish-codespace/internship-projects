"""
Voice Age & Emotion Detector — Hybrid (classical gender gate + deep age/emotion)
================================================================================
Pipeline:
  1. Gender gate -> trained classical Random Forest (gender_model.joblib)
  2. Age (male only) -> pretrained wav2vec2 (audEERING), outputs age in years
  3. Emotion (seniors) -> pretrained wav2vec2 (audEERING), arousal/dominance/valence

Needs: gender_model.joblib, gender_scaler.joblib in the same folder.
wav2vec2 models download from Hugging Face on first run.

Run with:
  pip install streamlit torch transformers==4.40.2 librosa joblib scikit-learn numpy
  streamlit run app2.py
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

# =======================================================================
# CONFIG
# =======================================================================
GENDER_MODEL_PATH = "gender_model.joblib"
GENDER_SCALER_PATH = "gender_scaler.joblib"
N_MFCC = 13
SAMPLE_RATE = 16000
SENIOR_THRESHOLD = 52

AGE_MODEL_NAME = "audeering/wav2vec2-large-robust-6-ft-age-gender"
EMO_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =======================================================================
# DEEP MODEL DEFINITIONS  (unchanged)
# =======================================================================
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
    def forward(self, x):
        x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
        return self.out_proj(x)

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0].mean(dim=1)
        return self.age(hidden), self.gender(hidden)

class EmotionRegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, x):
        x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
        return self.out_proj(x)

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = EmotionRegressionHead(config)
        self.init_weights()
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0].mean(dim=1)
        return self.classifier(hidden)


# =======================================================================
# LOADING + INFERENCE  (unchanged logic)
# =======================================================================
@st.cache_resource
def load_all():
    gender_clf = joblib.load(GENDER_MODEL_PATH)
    gender_scaler = joblib.load(GENDER_SCALER_PATH)
    age_proc = Wav2Vec2Processor.from_pretrained(AGE_MODEL_NAME)
    age_model = AgeGenderModel.from_pretrained(AGE_MODEL_NAME).to(device).eval()
    emo_proc = Wav2Vec2Processor.from_pretrained(EMO_MODEL_NAME)
    emo_model = EmotionModel.from_pretrained(EMO_MODEL_NAME).to(device).eval()
    return gender_clf, gender_scaler, age_proc, age_model, emo_proc, emo_model


def extract_gender_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if y.size == 0 or np.max(np.abs(y)) < 1e-4:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0_clean = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0_clean) if f0_clean.size > 0 else 0.0
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    sb = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.concatenate([mfcc_mean, [f0_mean, sc, sb, zcr]])


def predict_age_years(file_path, proc, model):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    inp = proc(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        age_logit, _ = model(inp)
    return float(age_logit[0][0]) * 100


def predict_emotion(file_path, proc, model):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    inp = proc(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        out = model(inp)
    arousal, dominance, valence = [float(v) for v in out[0]]
    if valence < 0.4 and arousal > 0.55:
        label = "Angry"
    elif valence < 0.45 and arousal <= 0.55:
        label = "Sad"
    elif valence >= 0.55:
        label = "Happy"
    else:
        label = "Neutral"
    return label, {"arousal": arousal, "dominance": dominance, "valence": valence}


def save_temp_audio(audio_obj):
    name = getattr(audio_obj, "name", None)
    suffix = os.path.splitext(name)[1] if name else ""
    if not suffix:
        suffix = ".wav"
    data = audio_obj.getvalue() if hasattr(audio_obj, "getvalue") else audio_obj.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


EMOTION_META = {
    "Angry":   {"emoji": "\U0001F525", "color": "#FF3B6B", "glow": "255,59,107"},
    "Sad":     {"emoji": "\U0001F4A7", "color": "#3B9BFF", "glow": "59,155,255"},
    "Happy":   {"emoji": "\u2728",      "color": "#FFC53B", "glow": "255,197,59"},
    "Neutral": {"emoji": "\U0001F300", "color": "#9B8BFF", "glow": "155,139,255"},
}


# =======================================================================
# PAGE + STYLE
# =======================================================================
st.set_page_config(page_title="VOX \u00b7 Neural Voice Analysis", page_icon="\U0001F9EC", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#05060B; --panel:rgba(18,22,38,0.55); --line:rgba(120,160,255,0.14);
  --cyan:#22E1FF; --magenta:#FF38C2; --violet:#8A6CFF; --lime:#9CFF4F;
  --ink:#EAF2FF; --muted:#8AA0C8; --muted2:#5566AA;
}
* { box-sizing:border-box; }

.stApp{
  background:
    radial-gradient(900px 600px at 12% 8%, rgba(34,225,255,0.10), transparent 55%),
    radial-gradient(900px 600px at 88% 92%, rgba(255,56,194,0.10), transparent 55%),
    radial-gradient(700px 500px at 70% 20%, rgba(138,108,255,0.08), transparent 60%),
    #05060B;
  background-attachment: fixed;
}
/* animated grid overlay */
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(120,160,255,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,160,255,0.045) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(ellipse 80% 70% at 50% 40%, #000 40%, transparent 100%);
}
.block-container{ padding:1.4rem 2.4rem 3rem; max-width:1500px; position:relative; z-index:1; }
#MainMenu, footer, header{ visibility:hidden; }

@keyframes floaty { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
@keyframes pulseGlow { 0%,100%{opacity:.55} 50%{opacity:1} }
@keyframes scan { 0%{background-position:0 -100%} 100%{background-position:0 200%} }
@keyframes sweep { 0%{transform:translateX(-120%)} 100%{transform:translateX(220%)} }

/* ---------- HERO ---------- */
.vox-hero{
  position:relative; border:1px solid var(--line); border-radius:22px;
  padding:30px 34px; margin-bottom:20px; overflow:hidden;
  background:
    linear-gradient(135deg, rgba(34,225,255,0.10), rgba(255,56,194,0.08) 60%, rgba(138,108,255,0.10)),
    rgba(10,14,26,0.6);
  backdrop-filter: blur(14px);
}
.vox-hero::after{
  content:""; position:absolute; top:0; left:0; width:40%; height:100%;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
  animation:sweep 6s linear infinite;
}
.vox-hero-row{ display:flex; align-items:center; gap:20px; position:relative; z-index:2; }
.vox-orb{
  width:66px; height:66px; border-radius:20px; flex-shrink:0;
  background:linear-gradient(145deg, var(--cyan), var(--violet) 55%, var(--magenta));
  display:flex; align-items:center; justify-content:center; font-size:32px;
  box-shadow:0 0 30px rgba(34,225,255,0.45), inset 0 0 18px rgba(255,255,255,0.25);
  animation:floaty 4s ease-in-out infinite;
}
.vox-name{
  font-family:'Orbitron',sans-serif; font-weight:900; font-size:42px; line-height:1;
  letter-spacing:4px;
  background:linear-gradient(90deg,#fff,var(--cyan) 40%,var(--magenta));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.vox-tag{
  font-family:'Rajdhani',sans-serif; font-weight:600; color:var(--muted);
  font-size:15px; letter-spacing:3px; text-transform:uppercase; margin-top:6px;
}
.vox-badges{ display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; position:relative; z-index:2; }
.vox-badge{
  font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--cyan);
  border:1px solid rgba(34,225,255,0.30); background:rgba(34,225,255,0.06);
  padding:4px 11px; border-radius:7px; letter-spacing:1px;
}
.vox-badge.m{ color:var(--magenta); border-color:rgba(255,56,194,0.30); background:rgba(255,56,194,0.06); }
.vox-badge.v{ color:var(--violet); border-color:rgba(138,108,255,0.30); background:rgba(138,108,255,0.06); }

/* ---------- PANEL ---------- */
.vox-panel{
  border:1px solid var(--line); border-radius:20px; padding:24px 26px;
  background:var(--panel); backdrop-filter:blur(16px);
  box-shadow:0 10px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
  height:100%;
}
.vox-eyebrow{
  font-family:'Orbitron',sans-serif; font-size:11px; font-weight:700; letter-spacing:3px;
  text-transform:uppercase; color:var(--cyan); margin-bottom:4px;
  display:flex; align-items:center; gap:8px;
}
.vox-eyebrow::before{ content:""; width:7px; height:7px; border-radius:50%;
  background:var(--cyan); box-shadow:0 0 10px var(--cyan); animation:pulseGlow 1.8s infinite; }
.vox-hint{ font-family:'Rajdhani',sans-serif; font-size:15px; color:var(--muted); margin-bottom:14px; font-weight:500; }

/* pipeline rail */
.vox-rail{ display:flex; flex-direction:column; gap:0; margin-top:6px; }
.vox-node{ display:flex; align-items:flex-start; gap:14px; position:relative; padding-bottom:18px; }
.vox-node:last-child{ padding-bottom:0; }
.vox-node::before{
  content:""; position:absolute; left:15px; top:32px; bottom:-2px; width:2px;
  background:linear-gradient(var(--line), transparent);
}
.vox-node:last-child::before{ display:none; }
.vox-dot{
  width:32px; height:32px; border-radius:10px; flex-shrink:0; z-index:1;
  display:flex; align-items:center; justify-content:center;
  font-family:'Orbitron',sans-serif; font-weight:700; font-size:13px; color:#05060B;
}
.vox-d1{ background:linear-gradient(145deg,var(--cyan),#1aa) ; box-shadow:0 0 14px rgba(34,225,255,0.5); }
.vox-d2{ background:linear-gradient(145deg,var(--violet),#65c); box-shadow:0 0 14px rgba(138,108,255,0.5); }
.vox-d3{ background:linear-gradient(145deg,var(--magenta),#a27); box-shadow:0 0 14px rgba(255,56,194,0.5); }
.vox-node-t{ font-family:'Rajdhani',sans-serif; font-weight:700; font-size:16px; color:var(--ink); }
.vox-node-d{ font-family:'Rajdhani',sans-serif; font-size:13.5px; color:var(--muted); }

/* ---------- RESULT CARDS ---------- */
.vox-card{
  border:1px solid var(--line); border-radius:16px; padding:18px 20px; margin-bottom:14px;
  background:rgba(12,16,30,0.5); position:relative; overflow:hidden;
}
.vox-card-top{ display:flex; align-items:center; justify-content:space-between; }
.vox-clabel{ font-family:'Orbitron',sans-serif; font-size:10.5px; font-weight:700; letter-spacing:2.5px;
  text-transform:uppercase; color:var(--muted); }
.vox-chip{ display:inline-flex; align-items:center; gap:7px; font-family:'Rajdhani',sans-serif;
  font-weight:700; font-size:15px; padding:6px 14px; border-radius:10px; }
.vox-big{ font-family:'Orbitron',sans-serif; font-weight:800; color:var(--ink); font-size:54px;
  line-height:1.05; margin-top:6px; letter-spacing:1px; }
.vox-big small{ font-family:'Rajdhani',sans-serif; font-size:20px; color:var(--muted); font-weight:600; }
.vox-cmeta{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted2,#667); margin-top:8px; letter-spacing:.5px; }

.vox-bar{ height:8px; border-radius:99px; background:rgba(255,255,255,0.05);
  border:1px solid var(--line); overflow:hidden; margin-top:12px; }
.vox-bar > div{ height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--cyan),var(--violet)); box-shadow:0 0 12px rgba(34,225,255,0.5); }

.vox-dimrow{ display:flex; align-items:center; gap:12px; margin:10px 0; }
.vox-dimn{ font-family:'Rajdhani',sans-serif; font-size:13px; color:var(--muted); width:88px;
  text-transform:uppercase; letter-spacing:1px; font-weight:600; }
.vox-dimt{ flex:1; height:7px; border-radius:99px; background:rgba(255,255,255,0.05);
  border:1px solid var(--line); overflow:hidden; }
.vox-dimf{ height:100%; border-radius:99px; }
.vox-dimv{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--ink); width:38px; text-align:right; }

.vox-senior{ display:inline-flex; align-items:center; gap:8px; font-family:'Orbitron',sans-serif;
  font-weight:700; font-size:12px; letter-spacing:1px; color:#FFC53B;
  background:rgba(255,197,59,0.10); border:1px solid rgba(255,197,59,0.35);
  border-radius:99px; padding:6px 14px; margin-bottom:10px;
  box-shadow:0 0 18px rgba(255,197,59,0.15); }

.vox-reject{ border:1px solid rgba(255,59,107,0.4); border-radius:16px; padding:22px 24px;
  background:rgba(255,59,107,0.08); display:flex; gap:16px; align-items:center;
  box-shadow:0 0 30px rgba(255,59,107,0.12); }
.vox-reject .ic{ font-size:30px; }
.vox-reject .t{ font-family:'Orbitron',sans-serif; font-weight:700; color:#FF7A9B; font-size:18px; letter-spacing:1px; }
.vox-reject .s{ font-family:'Rajdhani',sans-serif; color:var(--muted); font-size:14px; margin-top:3px; font-weight:500; }

.vox-empty{ text-align:center; padding:40px 20px; }
.vox-empty .ic{ font-size:46px; animation:floaty 4s ease-in-out infinite; }
.vox-empty .t{ font-family:'Orbitron',sans-serif; color:var(--muted); font-size:14px;
  letter-spacing:2px; margin-top:12px; text-transform:uppercase; }
.vox-empty .s{ font-family:'Rajdhani',sans-serif; color:var(--muted2,#667); font-size:14px; margin-top:6px; }

/* tabs + audio */
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{ font-family:'Rajdhani',sans-serif; font-weight:700; font-size:15px;
  letter-spacing:1px; color:var(--muted); background:transparent; padding:8px 18px; border-radius:9px; }
.stTabs [aria-selected="true"]{ color:var(--cyan)!important; background:rgba(34,225,255,0.07)!important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--cyan)!important; }

.vox-foot{ font-family:'JetBrains Mono',monospace; font-size:11px; color:#5566aa;
  text-align:center; margin-top:26px; letter-spacing:.5px; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="vox-hero">
  <div class="vox-hero-row">
    <div class="vox-orb">\U0001F9EC</div>
    <div>
      <div class="vox-name">VOX</div>
      <div class="vox-tag">Neural Voice Analysis Engine</div>
    </div>
  </div>
  <div class="vox-badges">
    <span class="vox-badge">GENDER \u00b7 RANDOM FOREST</span>
    <span class="vox-badge v">AGE \u00b7 WAV2VEC2</span>
    <span class="vox-badge m">EMOTION \u00b7 WAV2VEC2</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- load models ----------
try:
    gender_clf, gender_scaler, age_proc, age_model, emo_proc, emo_model = load_all()
except FileNotFoundError as e:
    st.markdown(f"""
    <div class="vox-reject">
      <div class="ic">\u26A0\uFE0F</div>
      <div><div class="t">MODEL FILES MISSING</div>
      <div class="s">Place gender_model.joblib and gender_scaler.joblib beside this script. ({e})</div></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ---------- two-column layout ----------
left, right = st.columns([1, 1.05], gap="large")

with left:
    st.markdown('<div class="vox-panel">', unsafe_allow_html=True)
    st.markdown('<div class="vox-eyebrow">Input Signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="vox-hint">Record a few seconds of speech or upload a clip. Male voices only \u2014 female input is returned.</div>', unsafe_allow_html=True)

    tab_record, tab_upload = st.tabs(["\U0001F3A4 Record", "\U0001F4C1 Upload"])
    audio_source = None
    with tab_record:
        rec = st.audio_input("Record a voice message")
        if rec is not None:
            audio_source = rec
    with tab_upload:
        up = st.file_uploader("Drop a .wav or .mp3", type=["wav", "mp3"])
        if up is not None:
            audio_source = up

    if audio_source is not None:
        st.audio(audio_source)

    # pipeline rail
    st.markdown("""
    <div style="margin-top:18px;"></div>
    <div class="vox-eyebrow">Pipeline</div>
    <div class="vox-rail">
      <div class="vox-node"><div class="vox-dot vox-d1">01</div>
        <div><div class="vox-node-t">Gender gate</div><div class="vox-node-d">Random Forest \u00b7 rejects female voices</div></div></div>
      <div class="vox-node"><div class="vox-dot vox-d2">02</div>
        <div><div class="vox-node-t">Age estimate</div><div class="vox-node-d">wav2vec2 transformer \u00b7 age in years</div></div></div>
      <div class="vox-node"><div class="vox-dot vox-d3">03</div>
        <div><div class="vox-node-t">Emotion scan</div><div class="vox-node-d">runs only for senior (60+) voices</div></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="vox-panel">', unsafe_allow_html=True)
    st.markdown('<div class="vox-eyebrow">Analysis Output</div>', unsafe_allow_html=True)

    if audio_source is None:
        st.markdown("""
        <div class="vox-empty">
          <div class="ic">\U0001F4E1</div>
          <div class="t">Awaiting signal</div>
          <div class="s">Record or upload a voice clip to begin analysis.</div>
        </div>""", unsafe_allow_html=True)
    else:
        tmp_path = save_temp_audio(audio_source)
        with st.spinner("Running neural analysis \u2014 deep models take a few seconds on CPU\u2026"):
            gfeats = extract_gender_features(tmp_path)
            gender_pred, gender_conf = None, 0.0
            if gfeats is not None:
                gs = gender_scaler.transform(gfeats.reshape(1, -1))
                gender_pred = gender_clf.predict(gs)[0]
                gender_conf = float(np.max(gender_clf.predict_proba(gs)[0]))
            result = {}
            if gfeats is not None and gender_pred == "male":
                age_years = predict_age_years(tmp_path, age_proc, age_model)
                result["age"] = age_years
                if age_years >= SENIOR_THRESHOLD:
                    result["emotion"] = predict_emotion(tmp_path, emo_proc, emo_model)
        os.remove(tmp_path)

        if gfeats is None:
            st.markdown("""
            <div class="vox-reject"><div class="ic">\U0001F507</div>
            <div><div class="t">SIGNAL UNREADABLE</div>
            <div class="s">Clip may be silent or too short. Try a longer, louder recording.</div></div></div>""",
            unsafe_allow_html=True)
        elif gender_pred == "female":
            st.markdown("""
            <div class="vox-reject"><div class="ic">\U0001F6AB</div>
            <div><div class="t">UPLOAD MALE VOICE</div>
            <div class="s">A female voice was detected. This system analyzes male voices only.</div></div></div>""",
            unsafe_allow_html=True)
        else:
            gconf = gender_conf * 100
            st.markdown(f"""
            <div class="vox-card">
              <div class="vox-card-top">
                <span class="vox-clabel">Gender</span>
                <span class="vox-chip" style="color:var(--cyan);background:rgba(34,225,255,0.10);border:1px solid rgba(34,225,255,0.35);">\u2713 MALE</span>
              </div>
              <div class="vox-bar"><div style="width:{gconf:.0f}%;"></div></div>
              <div class="vox-cmeta">CONFIDENCE {gconf:.1f}% \u00b7 RANDOM FOREST</div>
            </div>""", unsafe_allow_html=True)

            age_years = result["age"]
            if age_years >= SENIOR_THRESHOLD:
                label, dims = result["emotion"]
                meta = EMOTION_META.get(label, EMOTION_META["Neutral"])
                st.markdown(f"""
                <div class="vox-card">
                  <div class="vox-senior">\u2605 SENIOR CITIZEN</div>
                  <div class="vox-clabel">Estimated age</div>
                  <div class="vox-big">~{age_years:.0f}<small> yrs</small></div>
                  <div class="vox-cmeta">WAV2VEC2 \u00b7 SENIOR THRESHOLD {SENIOR_THRESHOLD}+</div>
                </div>""", unsafe_allow_html=True)

                dim_html = ""
                for k in ["arousal", "dominance", "valence"]:
                    v = dims[k]
                    dim_html += f"""<div class="vox-dimrow"><div class="vox-dimn">{k}</div>
                      <div class="vox-dimt"><div class="vox-dimf" style="width:{v*100:.0f}%;background:{meta['color']};box-shadow:0 0 10px rgba({meta['glow']},0.6);"></div></div>
                      <div class="vox-dimv">{v:.2f}</div></div>"""
                st.markdown(f"""
                <div class="vox-card">
                  <div class="vox-card-top">
                    <span class="vox-clabel">Emotion</span>
                    <span class="vox-chip" style="color:{meta['color']};background:rgba({meta['glow']},0.10);border:1px solid rgba({meta['glow']},0.40);box-shadow:0 0 20px rgba({meta['glow']},0.20);">{meta['emoji']} {label.upper()}</span>
                  </div>
                  <div style="margin-top:8px;">{dim_html}</div>
                  <div class="vox-cmeta">DIMENSIONAL MODEL \u00b7 AROUSAL / DOMINANCE / VALENCE</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="vox-card">
                  <div class="vox-clabel">Estimated age</div>
                  <div class="vox-big">~{age_years:.0f}<small> yrs</small></div>
                  <div class="vox-cmeta">WAV2VEC2 \u00b7 BELOW SENIOR THRESHOLD \u00b7 EMOTION SKIPPED</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- footer ----------
st.markdown(f"""
<div class="vox-foot">
  GENDER \u00b7 SELF-BUILT RANDOM FOREST &nbsp;//&nbsp; AGE &amp; EMOTION \u00b7 PRETRAINED WAV2VEC2 (AUDEERING)<br>
  COMPUTE: {device.upper()} \u00b7 FIRST RUN DOWNLOADS MODEL WEIGHTS
</div>
""", unsafe_allow_html=True)