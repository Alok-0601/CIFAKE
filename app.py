import os
import zipfile
import tempfile
import numpy as np
from PIL import Image
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAKE Detector",
    page_icon="🚫",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0d0d14; color: #e2e8f0; }

/* Hero title */
.hero-title {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 60%, #f0abfc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}
.badge {
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    border: 1px solid rgba(255,255,255,0.15);
}
.badge-purple { background: rgba(124,58,237,0.25); color: #c4b5fd; }
.badge-cyan   { background: rgba(6,182,212,0.2);   color: #67e8f9; }
.badge-green  { background: rgba(34,197,94,0.2);   color: #86efac; }

/* Result card */
.result-card {
    background: #13131f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    text-align: center;
}
.verdict-real { color: #22c55e; font-size: 1.6rem; font-weight: 900; }
.verdict-fake { color: #ef4444; font-size: 1.6rem; font-weight: 900; }
.confidence-label { color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem; }

/* Upload box */
.uploadedFile, [data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 2px dashed rgba(124,58,237,0.5) !important;
    border-radius: 14px !important;
}

/* Progress bar colours */
.stProgress > div > div { background: linear-gradient(90deg,#7c3aed,#06b6d4); }

/* Footer */
.footer {
    text-align: center;
    color: #4b5563;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.footer a { color: #a78bfa; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached — runs once per session) ────────────────────────────
@st.cache_resource(show_spinner="Loading model — please wait...")
def load_model():
    """Rebuild architecture from code + load weights. Bypasses ALL config issues."""
    from keras.applications import EfficientNetB0
    from keras import layers, Model

    base    = EfficientNetB0(weights=None, include_top=False, input_shape=(96, 96, 3))
    inputs  = layers.Input(shape=(96, 96, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1,   activation="sigmoid")(x)
    model   = Model(inputs, outputs)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile("efficientnet_model.keras", "r") as zf:
            zf.extractall(tmp)
        model.load_weights(os.path.join(tmp, "model.weights.h5"))

    return model


model    = load_model()
IMG_SIZE = (96, 96)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">CIFAKE Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Is that image real — or made by AI?</div>',
    unsafe_allow_html=True,
)
st.markdown("""
<div class="badge-row">
  <span class="badge badge-purple">⚡ EfficientNetB0</span>
  <span class="badge badge-cyan">🧠 Transfer Learning</span>
  <span class="badge badge-green">📊 CIFAKE Dataset</span>
  <span class="badge badge-purple">🖥️ Streamlit</span>
</div>
""", unsafe_allow_html=True)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True, caption="Uploaded image")

    with col_res:
        with st.spinner("Analysing..."):
            arr       = np.array(image.resize(IMG_SIZE), dtype=np.float32)[np.newaxis]
            prob_real = float(model.predict(arr, verbose=0)[0][0])
            prob_fake = 1.0 - prob_real

        label      = "REAL" if prob_real >= 0.5 else "FAKE"
        confidence = prob_real if label == "REAL" else prob_fake
        css_class  = "verdict-real" if label == "REAL" else "verdict-fake"
        icon       = "✅" if label == "REAL" else "🚨"

        st.markdown(f"""
        <div class="result-card">
          <div class="{css_class}">{icon} {label}</div>
          <div class="confidence-label">{confidence * 100:.1f}% confident</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.write("**Probability breakdown**")
        st.progress(prob_real, text=f"🟢 REAL — {prob_real*100:.1f}%")
        st.progress(prob_fake, text=f"🔴 FAKE — {prob_fake*100:.1f}%")

        with st.expander("ℹ️ Model details"):
            st.markdown("""
| Field | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Input size | 96 × 96 px |
| Output | Sigmoid (binary) |
| Dataset | CIFAKE (60 k images) |
            """)
else:
    st.markdown("""
    <div style="background:#13131f;border:2px dashed rgba(124,58,237,0.3);border-radius:14px;
                padding:2.5rem;text-align:center;color:#4b5563;margin-top:0.5rem;">
        📁 Drag &amp; drop an image above, or click to browse
    </div>
    """, unsafe_allow_html=True)


# ── How it works ─────────────────────────────────────────────────────────────
st.write("")
with st.expander("⚙️ How it works"):
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📤", "Upload",     "Any JPG / PNG / WEBP up to 200 MB"),
        (c2, "🔬", "Pre-process","Resized to 96 × 96 px"),
        (c3, "🧠", "Analyse",    "EfficientNetB0 extracts deep features"),
        (c4, "✅", "Verdict",    "REAL / FAKE + confidence score"),
    ]:
        with col:
            st.markdown(f"**{icon} {title}**\n\n{desc}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with ❤️ by <strong>Alok Gupta</strong> · ADGITM &nbsp;|&nbsp;
  Dataset: <a href="https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
              target="_blank">CIFAKE on Kaggle</a>
</div>
""", unsafe_allow_html=True)
