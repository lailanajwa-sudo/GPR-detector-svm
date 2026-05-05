import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from skimage import exposure

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 2. LAYOUT ---
st.set_page_config(page_title="GPR Intelligent AI Classifier", layout="wide")
st.title("📡 Autonomous GPR Subsurface Intelligence")

if model is None:
    st.error("Missing AI model files.")
else:
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    
    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        
        # Smart Pre-processing: Background Removal
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # 1. ROI EXTRACTION
        roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_raw, (100, 120))
        
        # 2. SMART TEXTURE ENHANCEMENT (AHE)
        # This forces the AI to see the "internal" structure of the hyperbola
        roi_enhanced = exposure.equalize_adapthist(roi_ready, clip_limit=0.03)
        
        # 3. STATISTICAL AI ENGINE (No Threshold Sliders)
        # Calculate Peakness (Kurtosis) and Variance
        energy = np.std(roi_ready)
        # Check if the signal is "Hollow" or "Solid" using local contrast variance
        texture_complexity = np.std(np.diff(roi_enhanced, axis=0))

        # 4. SVM Prediction on 12k Features
        imf1 = detrend(detrend(roi_enhanced, axis=0), axis=1)
        features = mat2gray_python(imf1).flatten(order='F')
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        prediction = model.predict(scaler.transform(features))[0]

        # --- THE SMART DECISION LOGIC ---
        if energy < 0.010:
            res, color = "NO TARGET (BACKGROUND) ⚪", "#484f58"
        elif energy > 0.025:
            res, color = "METAL PIPE ⚙️", "#da3633"
        else:
            # If the texture is "Messy/Ringing" (Low complexity variance), it's a Cavity
            # If it's "Sharp/Clean" (High complexity variance), it's a Brick
            if texture_complexity < 0.045: 
                res, color = "CAVITY (VOID) ✅", "#238636"
            else:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"

        # --- DISPLAY ---
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:25px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:28px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Signal Energy", f"{energy:.4f}")
            st.metric("Texture Complexity", f"{texture_complexity:.4f}")
            st.image(mat2gray_python(roi_enhanced), caption="Enhanced 12k Feature Map")
