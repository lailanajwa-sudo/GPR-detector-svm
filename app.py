import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl')
    except: return None, None

model, scaler = load_assets()

# --- PREPROCESSING WITH CONTRAST BOOST ---
def preprocess_roi(roi_img, contrast_val):
    # 1. Boost contrast to help find faint Brick/Cavity signals
    enhancer = ImageEnhance.Contrast(roi_img)
    roi_img = enhancer.enhance(contrast_val)
    
    # 2. Convert and resize
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # 3. Standardize (mat2gray)
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    return scaler.transform(img_norm.flatten().reshape(1, -1))

st.title("📡 GPR Precision Classifier")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    
    # CONTROLS
    st.write("### Target Settings")
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("X Position", 0, img.size[0]-80, int(img.size[0]/2))
        y = st.slider("Y Position", 0, img.size[1]-60, int(img.size[1]/2))
    with col2:
        # NEW: Sensitivity control
        gain = st.slider("Signal Gain (Contrast)", 0.5, 5.0, 1.5, help="Increase for Brick/Cavity")
    
    # Show the "AI View" immediately so you know if the pattern is clear
    roi = img.crop((x, y, x+80, y+60))
    st.image(roi.resize((240, 180)), caption="AI Preview (Zoomed)", width=200)

    if st.button("🚀 ANALYZE", use_container_width=True):
        features = preprocess_roi(roi, gain)
        probs = model.predict_proba(features)[0]
        classes = ["Cavity", "Brick", "Metal Pipe"]
        
        # Display Results
        res_idx = np.argmax(probs)
        st.subheader(f"Result: {classes[res_idx]}")
        
        # Show comparison
        cols = st.columns(3)
        for i, c in enumerate(classes):
            cols[i].metric(c, f"{probs[i]*100:.1f}%")
