import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('svm_model.pkl'), joblib.load('scaler.pkl')
    except: return None, None

model, scaler = load_assets()

# --- 2. ENHANCED PREPROCESSING (The "Anti-Cavity" Logic) ---
def preprocess_roi(roi_img, gain):
    # Convert to grayscale
    roi_gray = ImageOps.grayscale(roi_img)
    
    # STRETCH CONTRAST: This forces Bricks to stand out from the background
    # This prevents the AI from seeing "nothing" and guessing Cavity
    enhancer = ImageEnhance.Contrast(roi_gray)
    roi_gray = enhancer.enhance(gain)
    
    roi_resized = roi_gray.resize((120, 100))
    img_np = np.array(roi_resized).astype(np.float64)
    
    # Calculate local contrast (Standard Deviation)
    sig_std = np.std(img_np)
    
    # Global Normalization
    img_min, img_max = np.min(img_np), np.max(img_np)
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    
    features = scaler.transform(img_norm.flatten().reshape(1, -1))
    return features, sig_std

# --- 3. UI SETUP ---
st.set_page_config(layout="wide") 
st.title("📡 GPR Target Classifier (V4)")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60

    # Layout: Controls on the Left, HUGE Preview on the Right
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Controls")
        pos_x = st.slider("X Position (Horizontal)", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Y Position (Vertical)", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        st.subheader("Sensitivity Settings")
        gain = st.slider("Contrast Gain (Con)", 0.5, 5.0, 2.0, help="Boost this for Bricks")
        noise_filter = st.slider("Signal Threshold", 5.0, 50.0, 15.0)
        
        st.divider()
        # --- ANALYZE BUTTON AT THE BOTTOM OF SLIDERS ---
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")

    with col_prev:
        st.subheader("Radargram Preview")
        # Much larger figure for better visibility
        fig, ax = plt.subplots(figsize=(12, 7)) 
        ax.imshow(img)
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_btn:
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        features, strength = preprocess_roi(roi, gain)
        
        # Display Results in a clear box
        with st.container():
            st.divider()
            # If the area is too "flat", don't let it guess Cavity
            if strength < noise_filter:
                st.warning(f"⚠️ **Weak Signal Detected** ({strength:.1f}). This looks like background. Please move the box to a clearer hyperbola.")
            else:
                probs = model.predict_proba(features)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                res_idx = np.argmax(probs)
                
                st.header(f"Detection Result: {classes[res_idx]}")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Cavity Prob", f"{probs[0]*100:.1f}%")
                m2.metric("Brick Prob", f"{probs[1]*100:.1f}%")
                m3.metric("Metal Pipe Prob", f"{probs[2]*100:.1f}%")
                
                # Show the zoomed version
                st.write("### AI Processed View:")
                st.image(roi.resize((400, 300)), width=300)

elif not model:
    st.error("Missing model files (svm_model.pkl / scaler.pkl).")
