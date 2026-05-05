import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. UI SETUP ---
st.set_page_config(page_title="GPR Subsurface Classifier", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .result-card { 
        padding: 25px; border-radius: 15px; color: white; font-weight: bold; 
        text-align: center; font-size: 32px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
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

# --- 2. MAIN INTERFACE ---
st.title("📡 GPR Intelligent Subsurface Classifier")

if model is None:
    st.error("Model files not found!")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace", 0, 450-125, 200)
    # Recommending 0.007 to capture the faint Cavity
    soil_limit = st.sidebar.slider("Noise Filter", 0.001, 0.020, 0.007, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
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
            
            # --- SYMMETRY LOGIC ---
            left_half = roi_ready[:, :60]
            right_half = np.fliplr(roi_ready[:, 60:])
            # Symmetrical objects like Bricks/Pipes have high correlation
            symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]

            if energy < soil_limit:
                st.markdown('<div class="result-card" style="background-color: #484f58;">NO TARGET ⚪</div>', unsafe_allow_html=True)
            else:
                # RUN THE 12,000 FEATURE SVM AS A VOTE
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                f = mat2gray_python(imf1).flatten(order='F')
                f = np.pad(f, (0, 12000-len(f)))[:12000].reshape(1,-1)
                svm_vote = model.predict(scaler.transform(f))[0]

                # FINAL DECISION CALIBRATION
                if energy >= 0.026 and symmetry_score > 0.6:
                    res = "METAL PIPE ⚙️"
                    color = "#da3633"
                elif 0.013 <= energy < 0.026 and symmetry_score > 0.5:
                    res = "BRICK / CONCRETE 🧱"
                    color = "#d29922"
                else:
                    # If it's messy or lower energy, call it Cavity
                    res = "CAVITY (VOID) ✅"
                    color = "#238636"

                st.markdown(f'<div class="result-card" style="background-color: {color};">{res}</div>', unsafe_allow_html=True)

            st.metric("Reflection Energy", f"{energy:.4f}")
            st.metric("Symmetry Confidence", f"{symmetry_score:.2f}")
            st.image(mat2gray_python(roi_ready), caption="BEMD Processing ROI", use_container_width=True)
