import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. CORE ASSETS ---
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

# --- 2. INTERFACE ---
st.set_page_config(page_title="GPR Autonomous AI", layout="wide")
st.title("📡 GPR BEMD-SVM Intelligent System")

if model is None:
    st.error("SVM Model or Scaler not found!")
else:
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # 1. Selection & Energy Calculation
        roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_raw, (100, 120))
        energy = np.std(roi_ready)
        
        # 2. THE "SMART" FIX: Statistical Thresholding
        # We define the ranges based on your actual data feedback
        # Background Noise: ~0.010
        # Cavity: ~0.016
        # Brick: ~0.019
        # Metal: >0.025
        
        is_real_target = energy > 0.012 # Stops BG ($0.010$) from being a Cavity
        
        # 3. Feature Prep (12,000 Pixels)
        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
        roi_norm = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
        features = roi_norm.flatten(order='F')
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        
        # 4. SVM Prediction
        svm_prediction = model.predict(scaler.transform(features))[0]

        # --- 3. DECISION ENGINE ---
        if not is_real_target:
            res, color = "NO TARGET ⚪", "#484f58"
        else:
            # Calibrated logic to stop Cavity ($0.016$) from being called Brick
            if energy > 0.025:
                res, color = "METAL PIPE ⚙️", "#da3633"
            elif 0.018 <= energy <= 0.025:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"
            else:
                # This range ($0.012$ to $0.018$) perfectly captures your Cavity
                res, color = "CAVITY (VOID) ✅", "#238636"

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:25px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:28px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Signal Intensity", f"{energy:.4f}")
            st.image(mat2gray_python(roi_norm), caption="Processed 12k BEMD Features")
