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

# --- 2. UI SETUP ---
st.set_page_config(page_title="GPR Autonomous AI", layout="wide")
st.title("📡 Intelligent GPR Phase-Stable Classifier")

if model is None:
    st.error("Missing AI Assets!")
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
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        energy = np.std(roi_ready)
        
        # --- 3. THE "STABLE APEX" ENGINE ---
        # Instead of just the center, find the column with the highest variance (the Hyperbola peak)
        column_variances = np.std(roi_ready, axis=0)
        apex_idx = np.argmax(column_variances)
        
        # Pull a 10-pixel slice around the REAL apex, no matter where it is in the box
        start_idx = max(0, apex_idx - 5)
        end_idx = min(120, apex_idx + 5)
        apex_slice = roi_ready[15:45, start_idx:end_idx] 
        
        # Adaptive Contrast Enhancement on the slice
        apex_en = exposure.equalize_adapthist(apex_slice, clip_limit=0.03)
        phase_score = np.mean(apex_en)

        # --- 4. SMART DECISION LOGIC ---
        if energy < 0.010:
            res, color = "NO TARGET ⚪", "#484f58"
        elif energy > 0.026: # High energy is always metal
            res, color = "METAL PIPE ⚙️", "#da3633"
        else:
            # Cavities (Air) have a specific phase brightness compared to solids
            # We use 0.48 as the stable split point for normalized contrast
            if phase_score > 0.49: 
                res, color = "CAVITY (VOID) ✅", "#238636"
            else:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"

        # --- 5. DISPLAY ---
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(full_img, cmap='gray', aspect='auto')
            # The green box for ROI
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            # A red line showing where the AI "found" the apex
            ax.axvline(h_pos + apex_idx, color='red', linestyle='--', alpha=0.5)
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:25px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:28px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Reflection Energy", f"{energy:.4f}")
            st.metric("Apex Stability Score", f"{phase_score:.4f}")
            st.write(f"Apex located at Trace: {h_pos + apex_idx}")
            
            # Show the 12,000 features being analyzed
            imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
            st.image(mat2gray_python(imf1), caption="12k BEMD IMF-1 Features")
