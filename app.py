import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

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
st.set_page_config(page_title="GPR Intelligent Classifier", layout="wide")
st.title("📡 GPR BEMD-SVM Autonomous System")

if model is None:
    st.error("Missing svm_model.pkl or scaler.pkl")
else:
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)

    files = st.file_uploader("Upload Data", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        
        # --- THE SMART FEATURE ENGINE ---
        # 1. Isolate texture (BEMD IMF1 equivalent)
        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
        
        # 2. Z-Score Scaling: Makes faint Cavities and bright Metal look the same scale to the SVM
        roi_norm = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
        
        # 3. Geometry Check: Does it have a curved apex?
        # We check the variance of the center versus the whole box
        center_energy = np.std(roi_ready[:, 40:80])
        total_energy = np.std(roi_ready)
        is_curved = center_energy > (total_energy * 0.95) # Catches faint hyperbolas

        # 4. 12,000 Feature Prediction
        features = roi_norm.flatten(order='F')
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        prediction = model.predict(scaler.transform(features))[0]

        # --- 3. FINAL SMART CLASSIFICATION ---
        # If it's a flat soil bar (no curvature) or dead background, show nothing.
        if total_energy < 0.005:
            res, color = "SCANNING... ⚪", "#484f58"
        elif total_energy > 0.015 and not is_curved:
            # This ignores the horizontal soil layers at the top
            res, color = "SOIL LAYER (MEDIUM CHANGE) 🏜️", "#1a1c24"
        else:
            # Trust the SVM + Energy logic
            if total_energy > 0.023:
                res, color = "METAL PIPE ⚙️", "#da3633"
            elif 0.012 <= total_energy <= 0.023:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"
            else:
                # Faint but curved = CAVITY
                res, color = "CAVITY (VOID) ✅", "#238636"

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:24px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Intensity", f"{total_energy:.4f}")
            st.image(mat2gray_python(roi_norm), caption="12k Normalized Features")
