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
        m = joblib.load('svm_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 2. BEMD-STYLE PREPROCESSING (IMF-1 Extraction) ---
def apply_bemd_logic(roi_img):
    # Convert to grayscale and resize to training dimensions (120x100)
    roi_gray = np.array(ImageOps.grayscale(roi_img).resize((120, 100))).astype(float)
    
    # MIMIC BEMD: High-pass filter to extract the first IMF
    # This removes the background 'noise' and highlights the hyperbola curves
    low_freq = gaussian_filter(roi_gray, sigma=3)
    imf_1 = roi_gray - low_freq 
    
    # Fixed Normalization (mat2gray)
    f_min = np.min(imf_1)
    f_max = np.max(imf_1)
    
    # Prevent division by zero and normalize 0 to 1
    diff = f_max - f_min
    if diff < 1e-7:
        imf_norm = np.zeros_like(imf_1)
    else:
        imf_norm = (imf_1 - f_min) / diff
    
    return imf_norm

# --- 3. UI SETUP ---
st.set_page_config(layout="wide", page_title="GPR SVM-BEMD Classifier")
st.title("📡 SVM + BEMD GPR Classifier")

uploaded_file = st.file_uploader("Step 1: Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60 # Detection box size

    # Layout: Controls on the Left, Big Preview on the Right
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Positioning")
        pos_x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        # Analyze button at the bottom of the sliders
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")

    with col_prev:
        st.subheader("Radargram Preview")
        # Big Preview for precise selection
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION ---
    if analyze_btn:
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Apply the BEMD logic to get IMF-1
        processed_imf = apply_bemd_logic(roi)
        
        # Prepare for SVM (Flatten and use Scaler)
        features = scaler.transform(processed_imf.flatten().reshape(1, -1))
        probs = model.predict_proba(features)[0]
        
        # Class names (Must match your training order)
        classes = ["Cavity", "Brick", "Metal Pipe"]
        res_idx = np.argmax(probs)
        
        st.divider()
        st.header(f"Detection Result: {classes[res_idx]}")
        
        # Probabilities
        m1, m2, m3 = st.columns(3)
        m1.metric(classes[0], f"{probs[0]*100:.1f}%")
        m2.metric(classes[1], f"{probs[1]*100:.1f}%")
        m3.metric(classes[2], f"{probs[2]*100:.1f}%")
        
        # Show what the AI "sees" (The BEMD IMF)
        st.write("### AI Feature View (BEMD IMF-1):")
        # We use st.image and multiply by 255 to show the normalized IMF clearly
        st.image(processed_imf, width=400, clamp=True, caption="Pattern analyzed by SVM")

elif not model:
    st.error("Error: 'svm_model.pkl' or 'scaler.pkl' not found. Please upload them to your app folder.")
