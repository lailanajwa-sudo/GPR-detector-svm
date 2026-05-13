import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('svm_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 2. BEMD PREPROCESSING LOGIC ---
def apply_bemd_logic(roi_img):
    # Grayscale and Resize to match your 120x100 training data
    roi_gray = np.array(ImageOps.grayscale(roi_img).resize((120, 100))).astype(float)
    
    # BEMD IMF-1 Extraction (High-Pass Filter)
    # sigma=1.5 is better for detecting subtle Brick textures
    low_freq = gaussian_filter(roi_gray, sigma=1.5)
    imf_1 = roi_gray - low_freq 
    
    # Normalization (mat2gray)
    f_min, f_max = np.min(imf_1), np.max(imf_1)
    diff = f_max - f_min
    
    if diff < 1e-7:
        imf_norm = np.zeros_like(imf_1)
    else:
        imf_norm = (imf_1 - f_min) / diff
    
    return imf_norm

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide", page_title="GPR SVM-BEMD System")
st.title("📡 GPR Target Classifier (SVM + BEMD)")

uploaded_file = st.file_uploader("Upload Radargram Image", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60 # Box dimensions

    # Split screen: Controls (Left) | Preview (Right)
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Controls")
        pos_x = st.slider("Horizontal (X Position)", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Vertical (Y Position)", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        # Analyze button at the bottom of the sliders
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")

    with col_prev:
        st.subheader("Radargram Preview")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        # Red box to show ROI
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_btn:
        # Crop the 80x60 ROI from the original image
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Apply BEMD logic
        processed_imf = apply_bemd_logic(roi)
        
        # Format for SVM (1D array)
        features = scaler.transform(processed_imf.flatten().reshape(1, -1))
        
        # Get Probabilities
        probs = model.predict_proba(features)[0]
        classes = ["Cavity", "Brick", "Metal Pipe"]
        res_idx = np.argmax(probs)
        
        # Results Section
        st.divider()
        st.success(f"### DETECTION: {classes[res_idx]}")
        
        # Show Probabilities
        m1, m2, m3 = st.columns(3)
        m1.metric(classes[0], f"{probs[0]*100:.1f}%")
        m2.metric(classes[1], f"{probs[1]*100:.1f}%")
        m3.metric(classes[2], f"{probs[2]*100:.1f}%")
        
        # Show the AI View (BEMD Image)
        st.write("### AI Analysis View (BEMD IMF-1)")
        st.image(processed_imf, width=400, clamp=True, caption="Pattern used for classification")

elif not model:
    st.error("Error: Please upload 'svm_model.pkl' and 'scaler.pkl' to the app folder.")
