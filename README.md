# 🔍 CIFAKE Detector

A Streamlit web app that detects whether an image is **real** (camera-captured) or **AI-generated**, powered by a fine-tuned **EfficientNetB0** model trained on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

## 🚀 Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (include `efficientnet_model.keras`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Pick your repo, branch `main`, file `app.py`
4. Click **Deploy** — done! 🎉

> **Note:** The first cold start takes ~60 s while the model loads. Subsequent requests are instant.

## 📁 File structure

```
.
├── app.py                    # Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── efficientnet_model.keras  # Trained model weights (upload separately)
```

## 🛠️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Model details

| Field | Value |
|---|---|
| Architecture | EfficientNetB0 (fine-tuned) |
| Framework | Keras 3 / TensorFlow 2.16+ |
| Input size | 96 × 96 px |
| Output | Sigmoid — P(REAL) |
| Dataset | CIFAKE — 60 000 images |
| Classes | REAL, FAKE (AI-generated) |

## 👤 Author

**Alok Gupta** · ADGITM
