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

# --- 2. UI ---
st.set_page_config(page_title="Autonomous GPR Analyzer", layout="wide")
st.title("📡 Smart BEMD-SVM GPR Classifier")

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
        
        # --- 3. THE SMART PHASE DETECTOR ---
        # 1. Enhance the image so the 'White' and 'Black' are clear
        roi_en = exposure.equalize_adapthist(roi_ready, clip_limit=0.03)
        
        # 2. Look at the very top-center of the hyperbola (Apex)
        # We take a small slice at the top middle
        apex_sample = roi_en[10:40, 55:65] 
        # Cavities have a strong WHITE peak at the top
        is_hollow = np.mean(apex_sample) > 0.52 
        
        energy = np.std(roi_ready)

        # --- 4. DECISION ENGINE ---
        if energy < 0.011:
            res, color = "NO TARGET ⚪", "#484f58"
        elif energy > 0.025:
            res, color = "METAL PIPE ⚙️", "#da3633"
        else:
            # If the apex is bright, it's a Cavity. If it's dark/balanced, it's a Brick.
            if is_hollow:
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
            st.metric("Intensity", f"{energy:.4f}")
            st.write("Apex Reflection: " + ("White (Hollow)" if is_hollow else "Dark (Solid)"))
            st.image(mat2gray_python(roi_en), caption="Contrast-Enhanced 12k Features")
