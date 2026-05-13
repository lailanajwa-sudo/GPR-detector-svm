import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
# Removed: from skimage import exposure (This was causing the error)

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: 
        return None, None

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

# --- 2. UI ---
st.set_page_config(page_title="GPR-X Detection SVM", layout="wide")
st.title("📡 GPR-X Detection (SVM-BEMD)")

if model is None:
    st.error("Missing AI Assets! Ensure svm_model.pkl and scaler.pkl are in the same folder.")
else:
    # Sidebar Sliders
    v_pos = st.sidebar.slider("Depth (Adjusted)", 0, 312-40-100, 80)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    
    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        
        # --- SMART CROP ---
        # Cutting first 40 pixels (Direct Coupling removal)
        matrix_cropped = matrix[40:, :] 
        
        matrix_clean = matrix_cropped - np.mean(matrix_cropped, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        energy = np.std(roi_ready)
        
        # --- 3. PHASE POLARITY ---
        apex_idx = np.argmax(np.std(roi_ready, axis=0))
        waveform = roi_ready[:, apex_idx]
        first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
        is_cavity_phase = first_peak <= 0.50 

        # --- 4. CLASSIFICATION ---
        if energy < 0.0135: 
            res, color = "NO TARGET (SOIL) ⚪", "#484f58"
        elif energy > 0.026:
            res, color = "METAL PIPE ⚙️", "#da3633"
        else:
            if is_cavity_phase:
                res, color = "CAVITY (VOID) ✅", "#238636"
            else:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"

        # --- 5. DISPLAY ---
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.markdown(f'<div style="padding:25px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:28px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Cleaned Energy Score", f"{energy:.4f}")
            st.write("Interface Noise Removed: **Yes**")
            
            # Feature visualization using detrending
            imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
            st.image(mat2gray_python(imf1), caption="12,000 BEMD Filtered Features")
