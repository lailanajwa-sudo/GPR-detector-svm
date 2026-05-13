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

# --- 2. ENHANCED PREPROCESSING ---
def preprocess_roi(roi_img, gain):
    # Convert to grayscale
    roi_gray = ImageOps.grayscale(roi_img)
    
    # BRICK ENHANCER: This helps the AI see the "Con" (Contrast) of subtle targets
    enhancer = ImageEnhance.Contrast(roi_gray)
    roi_gray = enhancer.enhance(gain)
    
    # Resize to training dimensions
    roi_resized = roi_gray.resize((120, 100))
    img_np = np.array(roi_resized).astype(np.float64)
    
    # Check Signal-to-Noise Ratio (Standard Deviation)
    signal_strength = np.std(img_np)
    
    # Global Normalization
    img_min, img_max = np.min(img_np), np.max(img_np)
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    
    features = scaler.transform(img_norm.flatten().reshape(1, -1))
    return features, signal_strength

# --- 3. UI SETUP ---
st.set_page_config(layout="wide") # Use wide mode for bigger preview
st.title("📡 High-Resolution GPR Classifier")

# Always visible button
analyze_btn = st.button("🚀 ANALYZE TARGET", use_container_width=True, type="primary")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60

    # Layout: Controls on the Left, BIG Preview on the Right
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Adjust Target")
        x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
        y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
        gain = st.slider("Contrast Gain (Con)", 0.5, 5.0, 2.0, help="Higher gain helps find Bricks")
        noise_limit = st.slider("Min Signal Strength", 5, 50, 20)

    with col_prev:
        st.subheader("Big Preview (Red Box = AI View)")
        # Create a bigger plot
        fig, ax = plt.subplots(figsize=(10, 6)) # Increased figure size
        ax.imshow(img)
        rect = plt.Rectangle((x, y), BW, BH, linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION ---
    if analyze_btn:
        roi = img.crop((x, y, x + BW, y + BH))
        features, strength = preprocess_roi(roi, gain)
        
        st.divider()
        
        # If the background is too "flat", don't classify as Cavity
        if strength < noise_limit:
            st.error(f"❌ Low Signal Strength ({strength:.1f}). This looks like Background noise, not a target.")
        else:
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            res_idx = np.argmax(probs)
            
            # Show big result
            st.header(f"Detected: {classes[res_idx]}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Cavity", f"{probs[0]*100:.1f}%")
            c2.metric("Brick", f"{probs[1]*100:.1f}%")
            c3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
            
            # Show exactly what the AI processed
            st.write("### AI Zoom View (After Contrast Adjustment)")
            # Enlarging the ROI view for the user
            st.image(roi.resize((400, 300)), width=300)

elif not model:
    st.error("Model files not found.")
