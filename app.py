import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl')
    except: return None, None

model, scaler = load_assets()

# --- 2. BEMD-STYLE PREPROCESSING ---
def apply_bemd_logic(roi_img):
    # Convert to grayscale and resize
    roi_gray = np.array(ImageOps.grayscale(roi_img).resize((120, 100))).astype(float)
    
    # MIMIC BEMD: Subtract the mean (low frequency) to get the IMF (high frequency)
    # This removes the "flat" background that causes the Cavity bias
    low_freq = gaussian_filter(roi_gray, sigma=3)
    imf_1 = roi_gray - low_freq 
    
    # Normalize (mat2gray)
    img_min, img_max = np.min(imf_1), np.max(imf_1)
    imf_norm = (imf_1 - img_min) / (imf_max - img_min + 1e-7)
    
    return imf_norm

# --- 3. UI SETUP ---
st.set_page_config(layout="wide")
st.title("📡 SVM + BEMD GPR Classifier")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60

    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Positioning")
        pos_x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        # Analyze button directly under sliders
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")

    with col_prev:
        st.subheader("Radargram Preview")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION ---
    if analyze_btn:
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Apply the BEMD-style filter
        processed_img = apply_bemd_logic(roi)
        
        # Flatten and Scale for SVM
        features = scaler.transform(processed_img.flatten().reshape(1, -1))
        probs = model.predict_proba(features)[0]
        
        classes = ["Cavity", "Brick", "Metal Pipe"]
        res_idx = np.argmax(probs)
        
        st.divider()
        st.header(f"Detection Result: {classes[res_idx]}")
        
        # Display Probabilities
        m1, m2, m3 = st.columns(3)
        m1.metric("Cavity", f"{probs[0]*100:.1f}%")
        m2.metric("Brick", f"{probs[1]*100:.1f}%")
        m3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
        
        # Show what the AI "sees" after BEMD filter
        st.write("### AI View (BEMD IMF-1 Filter):")
        st.image(processed_img, width=300, caption="Filtered Hyperbola Area")

elif not model:
    st.error("Model files missing.")
