import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. UI & STYLING ---
st.set_page_config(page_title="GPR Autonomous Classifier", layout="wide")

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

# --- 2. THE CORE LOGIC ---
st.title("📡 GPR BEMD-SVM Intelligent Classifier")

if model is None:
    st.error("AI Model files (svm_model.pkl/scaler.pkl) not found!")
else:
    v_pos = st.sidebar.slider("Depth (Vertical)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450-125, 200)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # Extract ROI
        roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_raw, (100, 120))
        
        # --- SMART STEP 1: Feature Decoupling ---
        # Detrending isolates the BEMD IMF texture from the DC bias
        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
        
        # --- SMART STEP 2: Z-Score Normalization ---
        # This is the "Magic" - it strips the brightness so Brick and Metal look similar in scale
        # forcing the SVM to look at the CURVATURE of the 12,000 pixels.
        roi_norm = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
        
        # --- SMART STEP 3: Geometry Check ---
        # Horizontal bars (soil layers) have low variance across columns
        # Object hyperbolas have high variance.
        col_variance = np.var(roi_ready, axis=0)
        is_horizontal_layer = np.mean(col_variance) < 0.0005 
        energy = np.std(roi_ready)

        # Flatten for SVM (Order='F' for MATLAB compatibility)
        features = roi_norm.flatten(order='F')
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        
        # Predict using the 12k features
        prediction = model.predict(scaler.transform(features))[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#e74c3c', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            # Final Classification Display
            if energy < 0.009 or is_horizontal_layer:
                res, color = "NO TARGET ⚪", "#484f58"
            else:
                # Reality Check: If SVM says Metal but energy is low, it's a Brick/Cavity
                if prediction == 3 and energy < 0.022:
                    prediction = 2 # Downgrade to Brick
                
                if prediction == 1:
                    res, color = "CAVITY (VOID) ✅", "#238636"
                elif prediction == 2:
                    res, color = "BRICK / CONCRETE 🧱", "#d29922"
                else:
                    res, color = "METAL PIPE ⚙️", "#da3633"

            st.markdown(f'<div style="padding:30px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:26px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Reflection Energy", f"{energy:.4f}")
            st.image(mat2gray_python(roi_norm), caption="12,000 Normalized Features", use_container_width=True)
