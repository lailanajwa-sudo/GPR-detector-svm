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

# --- 2. ROBUST BEMD PREPROCESSING ---
def apply_bemd_logic(roi_img):
    # Convert to grayscale and resize to 120x100
    roi_gray = np.array(ImageOps.grayscale(roi_img).resize((120, 100))).astype(float)
    
    # SHARPENED BEMD: Sigma 1.2 captures fine brick textures better
    low_freq = gaussian_filter(roi_gray, sigma=1.2)
    imf_1 = roi_gray - low_freq 
    
    # ROBUST NORMALIZATION: Uses percentiles to ignore noise/outliers
    # This prevents the brick hyperbola from being 'washed out'
    f_min, f_max = np.percentile(imf_1, [1, 99]) 
    diff = f_max - f_min
    
    if diff < 1e-7:
        imf_norm = np.zeros_like(imf_1)
    else:
        # Normalize and clip values between 0 and 1
        imf_norm = np.clip((imf_1 - f_min) / diff, 0, 1)
    
    return imf_norm

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide", page_title="GPR SVM-BEMD Analysis")
st.title("📡 GPR Target Classifier (SVM + BEMD)")

uploaded_file = st.file_uploader("Upload Radargram Image", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60 # Fixed window for SVM features

    # Split screen: Controls on Left | BIG Preview on Right
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Positioning")
        # Sliders for target selection
        pos_x = st.slider("X Position (Horizontal)", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Y Position (Vertical)", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        # Analyze button positioned at the bottom of sliders
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")
        
        st.info("💡 Tip: For Bricks, ensure the box covers the top curve and the beginning of the 'legs'.")

    with col_prev:
        st.subheader("High-Resolution Radargram Preview")
        # Large plot for precise placement
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_btn:
        # Crop
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # BEMD Processing
        processed_imf = apply_bemd_logic(roi)
        
        # SVM Prediction
        features = scaler.transform(processed_imf.flatten().reshape(1, -1))
        probs = model.predict_proba(features)[0]
        
        # Mapping
        classes = ["Cavity", "Brick", "Metal Pipe"]
        res_idx = np.argmax(probs)
        
        st.divider()
        st.success(f"### DETECTION: {classes[res_idx]}")
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(classes[0], f"{probs[0]*100:.1f}%")
        m2.metric(classes[1], f"{probs[1]*100:.1f}%")
        m3.metric(classes[2], f"{probs[2]*100:.1f}%")
        
        # AI Debug View (The IMF-1 Feature)
        st.write("### AI Analysis View (BEMD IMF-1)")
        st.write("If you see a faint 'V' or texture here, the AI is correctly seeing the object.")
        st.image(processed_imf, width=400, clamp=True)

elif not model:
    st.error("Error: 'svm_model.pkl' or 'scaler.pkl' not found in the app directory.")
