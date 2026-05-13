import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('svm_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img, gain):
    # Enhance Contrast to bring out faint Brick legs
    enhancer = ImageEnhance.Contrast(roi_img)
    roi_img = enhancer.enhance(gain)
    
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # Calculate Signal Intensity (Standard Deviation)
    # This helps us identify if there is actually a "pattern" there
    intensity = np.std(img_np)
    
    # Normalization (0-1)
    img_min, img_max = np.min(img_np), np.max(img_np)
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    
    features = scaler.transform(img_norm.flatten().reshape(1, -1))
    return features, intensity

# --- 3. UI SETUP ---
st.set_page_config(layout="centered")
st.title("📡 GPR Pattern Classifier")

# Always visible Analyze Button
analyze_btn = st.button("🚀 ANALYZE SELECTED AREA", use_container_width=True, type="primary")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60

    st.write("### 1. Locate the Hyperbola")
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
        y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
    with col2:
        gain = st.slider("Signal Gain", 0.5, 5.0, 2.0) # Default to 2.0 for better Brick visibility

    # Small Preview
    _, mid_col, _ = st.columns([1, 4, 1])
    with mid_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(img)
        rect = patches.Rectangle((x, y), BW, BH, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_btn:
        roi = img.crop((x, y, x + BW, y + BH))
        features, sig_intensity = preprocess_roi(roi, gain)
        
        # Get Probabilities
        probs = model.predict_proba(features)[0]
        classes = ["Cavity", "Brick", "Metal Pipe"]
        
        # --- BIAS CORRECTION LOGIC ---
        # If the AI thinks it's a Cavity but the signal is strong, 
        # it might actually be a faint Brick.
        if np.argmax(probs) == 0 and sig_intensity > 35:
            # Re-evaluate: Look for the second best choice
            top_idx = 1 if probs[1] > probs[2] else 2
        else:
            top_idx = np.argmax(probs)

        # UI RESULTS
        st.divider()
        st.success(f"### RESULT: {classes[top_idx]}")
        
        # Show specific metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Cavity", f"{probs[0]*100:.1f}%")
        m2.metric("Brick", f"{probs[1]*100:.1f}%")
        m3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
        
        st.write(f"**Signal Intensity:** {sig_intensity:.2f}")
        st.image(roi.resize((240, 180)), caption="AI Analysis View", width=200)

elif not model:
    st.error("Missing model files.")
