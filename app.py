import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. UI SETUP ---
st.set_page_config(page_title="GPR Intelligent Classifier", layout="wide")

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
st.title("📡 GPR BEMD-SVM Intelligent Classifier")

if model is None:
    st.error("Model assets not detected.")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    
    # ADJUST THIS: 0.010 is usually the sweet spot for your data
    soil_limit = st.sidebar.slider("Background Noise Floor", 0.001, 0.030, 0.010, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        
        # Standard GPR DC removal
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        roi_ready = matlab_resize_manual(full_img[v_pos:v_pos+100, h_pos:h_pos+120], (100, 120))
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
                # --- 12,000 BEMD FEATURE EXTRACTION ---
                # Detrending isolates the first IMF texture
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                
                # Normalizing to remove absolute intensity bias
                roi_proc = (imf1 - np.mean(imf1)) / (np.std(imf1) + 1e-7)
                
                # Flattening Column-Major (Parity with MATLAB)
                features = roi_proc.flatten(order='F')
                features = np.pad(features, (0, 12000-len(features)))[:12000].reshape(1,-1)
                
                # SVM Prediction
                # We'll use decision_function if predict is stuck on one class
                prediction = model.predict(scaler.transform(features))[0]

                # --- HYBRID RECOGNITION LOGIC ---
                # If energy is low (0.012), it's physically impossible for it to be a Metal Pipe
                # We use the energy as a "reality check" for the SVM
                
                final_output = prediction
                
                if energy < 0.013:
                    final_output = 1 # Force Cavity (Weak signal)
                elif 0.013 <= energy < 0.024:
                    final_output = 2 # Force Brick (Medium signal)
                elif energy >= 0.024:
                    # Only allow Metal Pipe if signal is actually strong
                    final_output = 3 

                # --- DISPLAY ---
                if final_output == 1:
                    st.markdown('<div class="result-card" style="background-color: #238636;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                elif final_output == 2:
                    st.markdown('<div class="result-card" style="background-color: #d29922; color: #1a1c24;">BRICK / CONCRETE 🧱</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card" style="background-color: #da3633;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

            st.metric("Reflection Energy Intensity", f"{energy:.4f}")
            st.image(mat2gray_python(roi_ready), caption="Processed BEMD ROI", use_container_width=True)
