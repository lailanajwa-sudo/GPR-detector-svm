import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from scipy.stats import kurtosis

# --- 1. UI SETUP ---
st.set_page_config(page_title="GPR Intelligent AI Classifier", layout="wide")

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

# --- 2. MAIN APP ---
st.title("📡 GPR BEMD-SVM Autonomous Classifier")

if model is None:
    st.error("AI Model assets not found!")
else:
    st.sidebar.header("🕹️ Position Selection")
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)

    files = st.file_uploader("Upload GPR Data (.rad & .rd3)", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # 1. Extraction
        roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_raw, (100, 120))
        
        # 2. THE SMART FIX: Z-Score Normalization
        # This strips away the "brightness" that causes Metal-Pipe bias
        # and forces the SVM to see the SHAPE of the 12,000 features.
        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
        roi_norm = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
        
        # 3. STATISTICAL GUARD (Detects if it's a target or just flat soil)
        # Real hyperbolas have higher kurtosis (peakedness) than horizontal bars
        k_val = np.mean(kurtosis(roi_norm, axis=0))
        energy = np.std(roi_ready)
        is_target = k_val > -0.4 and energy > 0.009

        # 4. SVM PREDICTION
        features = roi_norm.flatten(order='F') # Exact MATLAB order
        features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
        
        # Predict using the 12,000 processed features
        prediction = model.predict(scaler.transform(features))[0]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("AI System Output")
            
            if not is_target:
                res, color = "NO TARGET (SOIL) ⚪", "#484f58"
            else:
                # Trust the SVM's decision on the normalized 12,000 features
                if prediction == 1:
                    res, color = "CAVITY (VOID) ✅", "#238636"
                elif prediction == 2:
                    res, color = "BRICK / CONCRETE 🧱", "#d29922"
                else:
                    res, color = "METAL PIPE ⚙️", "#da3633"

            st.markdown(f'<div style="padding:25px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:28px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Hyperbola Peakedness (Kurtosis)", f"{k_val:.2f}")
            st.metric("Signal Energy", f"{energy:.4f}")
            st.image(mat2gray_python(roi_norm), caption="12,000 Normalized BEMD Features", use_container_width=True)
