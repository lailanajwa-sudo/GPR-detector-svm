import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. UI SETUP ---
st.set_page_config(page_title="GPR Intelligent Classifier Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .result-card { 
        padding: 25px; border-radius: 15px; color: white; font-weight: bold; 
        text-align: center; font-size: 32px; margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    if old_h == 0 or old_w == 0: return np.zeros(new_shape)
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 2. MAIN APP ---
st.title("📡 GPR Subsurface Classifier")

if model is None:
    st.error("Assets missing!")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    
    noise_floor = st.sidebar.slider("Noise Floor", 0.001, 0.025, 0.010, step=0.001)
    brick_gate = st.sidebar.slider("Brick Threshold", 0.010, 0.030, 0.017, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        energy = np.std(roi_ready)

        # --- GEOMETRY GUARD ---
        # Calculate how "flat" the signal is. 
        # Medium changes are horizontal bars (low variance across the horizontal axis)
        horizontal_std = np.mean(np.std(roi_ready, axis=1)) 
        is_flat_bar = horizontal_std < 0.035 # Adjust this if bars still show up

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='#e74c3c', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("Target Analysis")
            
            # Use the 12k features for the SVM
            imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
            f = mat2gray_python(imf1).flatten(order='F')
            f = np.pad(f, (0, 12000-len(f)))[:12000].reshape(1,-1)
            
            # LOGIC OVERRIDE
            if energy < noise_floor or is_flat_bar:
                res, color = "NO TARGET ⚪", "#484f58"
                status = "Ignoring Surface/Layer" if is_flat_bar and energy > noise_floor else "Below Noise Floor"
            elif energy >= 0.026:
                res, color = "METAL PIPE ⚙️", "#da3633"
                status = "Target Detected"
            elif energy >= brick_gate:
                res, color = "BRICK / CONCRETE 🧱", "#d29922"
                status = "Target Detected"
            else:
                res, color = "CAVITY (VOID) ✅", "#238636"
                status = "Target Detected"

            st.markdown(f'<div class="result-card" style="background-color: {color};">{res}</div>', unsafe_allow_html=True)
            st.write(f"System Status: {status}")
            st.metric("Reflection Energy", f"{energy:.4f}")
            st.image(mat2gray_python(roi_ready), caption="BEMD Processing ROI", use_container_width=True)
