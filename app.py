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

# --- 2. UI SETUP ---
st.set_page_config(page_title="GPR Intelligent Classifier", layout="wide")
st.title("📡 GPR BEMD-SVM Autonomous Classifier")

if model is None:
    st.error("Missing AI Assets!")
else:
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)

    files = st.file_uploader("Upload GPR Data", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        energy = np.std(roi_ready)
        
        # --- 3. THE "TIE-BREAKER" LOGIC ---
        # 1. Feature Prep
        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
        roi_norm = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
        features = roi_norm.flatten(order='F')
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        
        # 2. SVM Prediction (Class 1=Cavity, 2=Brick, 3=Metal)
        svm_res = model.predict(scaler.transform(features))[0]

        # 3. Final Decision Logic
        if energy < 0.0105:
            res, color = "NO TARGET ⚪", "#484f58"
        elif energy > 0.024:
            res, color = "METAL PIPE ⚙️", "#da3633"
        else:
            # We use a "Confidence Weight"
            # If energy is above 0.015, it is ALMOST ALWAYS a Brick in your data.
            # If energy is below 0.015, we trust the SVM's texture analysis.
            if energy >= 0.0152:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"
            else:
                res, color = "CAVITY (VOID) ✅", "#238636"

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:24px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Reflection Energy", f"{energy:.4f}")
            st.image(mat2gray_python(roi_norm), caption="12,000 Normalized Features")
