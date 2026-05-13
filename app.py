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
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. IMPROVED BEMD PREPROCESSING ---
def apply_bemd_logic(roi_img):
    # Convert to grayscale and resize to training size (120x100)
    roi_gray = np.array(ImageOps.grayscale(roi_img).resize((120, 100))).astype(float)
    
    # SHARPENED BEMD: Sigma 1.5 preserves Brick textures better than Sigma 3.0
    # This extracts the 'high-frequency' details (the hyperbola edges)
    low_freq = gaussian_filter(roi_gray, sigma=1.5)
    imf_1 = roi_gray - low_freq 
    
    # ADVANCED NORMALIZATION: Stretch the signal so Bricks aren't 'faint'
    f_min = np.min(imf_1)
    f_max = np.max(imf_1)
    diff = f_max - f_min
    
    if diff < 1e-7:
        imf_norm = np.zeros_like(imf_1)
    else:
        # Standardize the signal to a 0.0 - 1.0 range
        imf_norm = (imf_1 - f_min) / diff
    
    return imf_norm

# --- 3. UI LAYOUT ---
st.set_page_config(layout="wide", page_title="GPR SVM-BEMD System")
st.title("📡 GPR Target Classifier (SVM + BEMD)")

uploaded_file = st.file_uploader("Step 1: Upload Radargram Image", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    BW, BH = 80, 60 # Detection window size

    # Layout: Controls on the left, Large Preview on the right
    col_ctrl, col_prev = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Manual Controls")
        pos_x = st.slider("Horizontal (X)", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Vertical (Y)", 0, max(1, h - BH), int(h/2))
        
        st.divider()
        # Analyze button positioned under sliders
        analyze_btn = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")

    with col_prev:
        st.subheader("High-Resolution Preview")
        # Larger figure size for precise target selection
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img)
        # Red rectangle to show current selection
        rect = plt.Rectangle((pos_x, pos_y), BW, BH, linewidth=4, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_btn:
        # Crop selected area
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Apply BEMD IMF extraction
        processed_imf = apply_bemd_logic(roi)
        
        # Format for SVM (Flattening 120x100 -> 12000 features)
        features = scaler.transform(processed_imf.flatten().reshape(1, -1))
        
        # Predict Probabilities
        probs = model.predict_proba(features)[0]
        classes = ["Cavity", "Brick", "Metal Pipe"]
        res_idx = np.argmax(probs)
        
        # Display Final Result
        st.divider()
        st.success(f"### RESULT: {classes[res_idx]}")
        
        # Breakdown Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(classes[0], f"{probs[0]*100:.1f}%")
        m2.metric(classes[1], f"{probs[1]*100:.1f}%")
        m3.metric(classes[2], f"{probs[2]*100:.1f}%")
        
        # AI Debug View
        st.write("### AI Analysis View (BEMD IMF-1 Result)")
        st.write("This is the pattern the AI sees. If you see a 'V' shape here, it is a valid target.")
        st.image(processed_imf, width=400, clamp=True)

elif not model:
    st.warning("⚠️ Waiting for 'svm_model.pkl' and 'scaler.pkl' to be placed in the directory.")
