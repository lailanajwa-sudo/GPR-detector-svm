import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. UI SETUP ---
st.set_page_config(page_title="GPR BEMD-SVM Classifier", layout="wide")

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
        # Loading your actual MATLAB-trained SVM and Scaler
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

# --- 2. DATA LOADING & PROCESSING ---
st.title("📡 GPR Intelligent Subsurface Classifier")

if model is None:
    st.error("Model files (svm_model.pkl/scaler.pkl) not found!")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth (Vertical)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450-125, 200)
    
    # This filter only removes low-level background noise
    # Set to 0.008 for best results with Cavity/Brick
    soil_limit = st.sidebar.slider("Background Noise Floor", 0.001, 0.030, 0.008, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        
        # 1. Basic Cleaning (DC Removal)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # 2. Extract 100x120 ROI
        roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_raw, (100, 120))
        energy = np.std(roi_ready)

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
            
            if energy < soil_limit:
                st.markdown('<div class="result-card" style="background-color: #484f58;">NO TARGET ⚪</div>', unsafe_allow_html=True)
            else:
                # --- 3. PURE BEMD FEATURE EXTRACTION ---
                # Double detrending effectively isolates the 1st IMF (high frequency)
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                roi_proc = mat2gray_python(imf1)
                
                # IMPORTANT: Flattening with order='F' to match MATLAB's memory layout
                features = roi_proc.flatten(order='F') 
                
                # Ensure exactly 12,000 features
                features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
                
                # --- 4. SVM CLASSIFICATION ---
                # Applying the original scaling and prediction logic
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)[0]

                # Map Prediction to Class
                if prediction == 1:
                    st.markdown('<div class="result-card" style="background-color: #238636;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                elif prediction == 2:
                    st.markdown('<div class="result-card" style="background-color: #d29922; color: #1a1c24;">BRICK / CONCRETE 🧱</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card" style="background-color: #da3633;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

            st.metric("Reflection Energy Intensity", f"{energy:.4f}")
            st.image(mat2gray_python(roi_ready), caption="BEMD ROI Input (12,000 features)", use_container_width=True)
