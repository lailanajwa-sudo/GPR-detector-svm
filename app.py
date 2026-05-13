import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl')
    except: return None, None

model, scaler = load_assets()

# --- 2. ADVANCED PREPROCESSING ---
def preprocess_roi(roi_img, gain):
    # Convert to grayscale first
    roi_gray = ImageOps.grayscale(roi_img)
    
    # BRICK BOOSTER: Contrast enhancement 
    # A gain of 2.0+ helps the AI see the "legs" of the brick hyperbola
    enhancer = ImageEnhance.Contrast(roi_gray)
    roi_gray = enhancer.enhance(gain)
    
    roi_resized = roi_gray.resize((120, 100))
    img_np = np.array(roi_resized).astype(np.float64)
    
    # CALCULATE SIGNAL STRENGTH (To distinguish Background vs. Objects)
    # Background usually has a very low standard deviation
    sig_std = np.std(img_np)
    
    # mat2gray Normalization
    img_min, img_max = np.min(img_np), np.max(img_np)
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    
    features = scaler.transform(img_norm.flatten().reshape(1, -1))
    return features, sig_std

# --- 3. UI SETUP ---
st.set_page_config(layout="centered")
st.title("📡 GPR Target Classifier (Pro)")

# Always visible button at the very top
analyze_btn = st.button("🚀 ANALYZE SELECTION", use_container_width=True, type="primary")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60

    # POSITIONING SLIDERS
    st.write("### Target Settings")
    col_pos, col_set = st.columns(2)
    with col_pos:
        x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
        y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
    with col_set:
        gain = st.slider("Signal Gain", 0.5, 5.0, 2.5) # Higher default for Bricks
        # This threshold prevents the "Background = Cavity" error
        noise_floor = st.slider("Background Filter", 10, 60, 25)

    # PREVIEW (Small and Centered)
    st.image(img.crop((x-50 if x>50 else 0, y-50 if y>50 else 0, x+BW+50, y+BH+50)), 
             caption="Area Preview", width=300)

    # --- 4. CLASSIFICATION ---
    if analyze_btn:
        roi = img.crop((x, y, x + BW, y + BH))
        features, strength = preprocess_roi(roi, gain)
        
        # 1. CHECK IF BACKGROUND
        if strength < noise_floor:
            st.warning(f"⚠️ **Background Detected** (Strength: {strength:.1f}). Move the box to a hyperbola.")
        else:
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            
            # 2. APPLY BIAS CORRECTION
            # If it's a weak signal, it's more likely a Cavity. 
            # If it's a medium signal, it's likely a Brick.
            # If it's a high signal, it's likely Metal.
            res_idx = np.argmax(probs)
            
            st.divider()
            st.success(f"### RESULT: {classes[res_idx]}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Cavity", f"{probs[0]*100:.1f}%")
            m2.metric("Brick", f"{probs[1]*100:.1f}%")
            m3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
            
            st.write(f"**Calculated Signal Strength:** {strength:.1f}")
            st.image(roi.resize((240, 180)), caption="AI Input View", width=150)
