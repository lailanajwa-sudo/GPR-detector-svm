import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="GPR Object Detector Pro", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; }
    .result-card { 
        padding: 25px; border-radius: 15px; color: white; font-weight: bold; 
        text-align: center; font-size: 28px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
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

# --- 2. PROCESSING ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    if old_h == 0 or old_w == 0: return np.zeros(new_shape)
    scale_y, scale_x = new_shape[0] / old_h, new_shape[1] / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_shape[0] + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_shape[1] + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 3. UI & LOGIC ---
st.title("📡 GPR Intelligent Subsurface Classifier")

if model is None:
    st.error("Missing .pkl files!")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 130)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    # SET THIS TO 0.012 for the demo!
    soil_limit = st.sidebar.slider("Noise Sensitivity", 0.005, 0.030, 0.012, step=0.001)

    files = st.file_uploader("Upload Data", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
        energy = np.std(roi)

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
            
            # --- ADVANCED APEX DETECTION LOGIC ---
            # 1. Check if the signal is just a flat horizontal line (Noise)
            horizontal_profile = np.mean(roi, axis=0)
            is_flat = np.std(horizontal_profile) < 0.05 # Low variance horizontally means it's a line
            
            # 2. Check if the energy is centered (Hyperbola Apex)
            center_zone = np.mean(roi[10:50, 40:80])
            side_zones = (np.mean(roi[10:50, :20]) + np.mean(roi[10:50, 100:])) / 2
            
            if energy < soil_limit or (is_flat and energy < 0.04):
                st.markdown('<div class="result-card" style="background-color: #484f58;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                st.write("Status: Scanning Ground Layers")
            else:
                # RUN SVM
                f = mat2gray_python(detrend(detrend(roi, axis=0), axis=1)).flatten(order='F')
                f = np.pad(f, (0, 12000-len(f)))[:12000].reshape(1,-1)
                pred = model.predict(scaler.transform(f))[0]
                
                # PHYSICAL OVERRIDES (Based on your screenshots)
                if energy > 0.032 and not is_flat: 
                    pred = 3 # Metal
                elif 0.015 < energy <= 0.032: 
                    pred = 2 # Brick/Concrete
                
                # DISPLAY
                if pred == 1: st.markdown('<div class="result-card" style="background-color: #238636;">CAVITY ✅</div>', unsafe_allow_html=True)
                elif pred == 2: st.markdown('<div class="result-card" style="background-color: #d29922; color: #0d1117;">BRICK / CONCRETE 🧱</div>', unsafe_allow_html=True)
                else: st.markdown('<div class="result-card" style="background-color: #da3633;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

            st.metric("Energy Intensity", f"{energy:.4f}")
            st.image(mat2gray_python(roi), caption="ROI Analysis Screen", use_container_width=True)
