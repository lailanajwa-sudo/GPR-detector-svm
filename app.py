import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl')
    except: return None, None

model, scaler = load_assets()

# --- PREPROCESSING ---
def preprocess_roi(roi_img, gain):
    enhancer = ImageEnhance.Contrast(roi_img)
    roi_img = enhancer.enhance(gain)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray normalization
    img_diff = np.max(img_np) - np.min(img_np)
    if img_diff < 1e-7: return None, 0 # Error handling for solid colors
    
    img_norm = (img_np - np.min(img_np)) / img_diff
    return scaler.transform(img_norm.flatten().reshape(1, -1)), img_diff

# --- UI ---
st.set_page_config(layout="centered")
st.title("📡 GPR Pattern Classifier (V3)")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # CONTROLS
    st.write("### Target Selection")
    c1, c2 = st.columns(2)
    with c1:
        x = st.slider("X Position", 0, w-80, int(w/2))
        y = st.slider("Y Position", 0, h-60, int(h/2))
    with c2:
        gain = st.slider("Sensitivity (Gain)", 0.5, 4.0, 1.0)
        # NEW: Sensitivity Threshold
        threshold = st.slider("Noise Filter", 5, 50, 20, help="Higher = stricter on what counts as a signal")

    # PREVIEW
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(img)
    rect = patches.Rectangle((x, y), 80, 60, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    if st.button("🚀 ANALYZE", use_container_width=True):
        roi = img.crop((x, y, x+80, y+60))
        features, raw_contrast = preprocess_roi(roi, gain)
        
        # LOGIC: If contrast is too low, it's NOT a cavity/brick/metal, it's just background.
        if raw_contrast < threshold:
            st.warning(f"⚠️ **Background Detected.** (Signal Strength: {raw_contrast:.1f}). Please move the box to a visible hyperbola.")
        else:
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            res = np.argmax(probs)
            
            # If the probability is too low, don't trust it
            if probs[res] < 0.45:
                st.info("❓ Uncertain Signal. Pattern does not clearly match known classes.")
            else:
                st.success(f"### Detected: {classes[res]}")
                st.write(f"Confidence: {probs[res]*100:.1f}%")
                
                cols = st.columns(3)
                for i, c in enumerate(classes):
                    cols[i].metric(c, f"{probs[i]*100:.1f}%")
