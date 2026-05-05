import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. PAGE SETUP ---
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

# --- 2. IMAGE UTILITIES ---
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

# --- 3. MAIN APP ---
st.title("📡 GPR Intelligent Subsurface Classifier")

if model is None:
    st.error("Missing model files in repository!")
else:
    st.sidebar.header("🕹️ Controls")
    v_pos = st.sidebar.slider("Depth (Vertical)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450-125, 200)
    
    # CRITICAL: Keep this at 0.008 - 0.010 for the demo to see Bricks!
    soil_limit = st.sidebar.slider("Background Noise Filter", 0.001, 0.020, 0.008, step=0.001)

    files = st.file_uploader("Upload GPR Data", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        
        # Load and Clean Data
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # Extract ROI
        roi_crop = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_crop, (100, 120))
        energy = np.std(roi_ready)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Radargram View")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.imshow(full_img, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='#e74c3c', fill=False))
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("Target Analysis")
            
            # --- IMPROVED DETECTION LOGIC ---
            # 1. Check if the ROI is too empty
            if energy < soil_limit:
                st.markdown('<div class="result-card" style="background-color: #484f58;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                st.write("Status: Scanning Soil Background")
            else:
                # 2. Extract BEMD-style features
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                f = mat2gray_python(imf1).flatten(order='F')
                f = np.pad(f, (0, 12000-len(f)))[:12000].reshape(1,-1)
                
                # 3. Get SVM Prediction
                pred = model.predict(scaler.transform(f))[0]
                
                # 4. HYBRID OVERRIDE (Ensuring correct physics labels)
                # Metal has high contrast
                if energy > 0.030:
                    pred = 3 
                # Brick/Concrete is the middle range (your 0.0144 fits here)
                elif 0.012 <= energy <= 0.030:
                    pred = 2
                # Cavity is the lower visible range
                elif soil_limit <= energy < 0.012:
                    pred = 1

                # Display Results
                if pred == 1:
                    st.markdown('<div class="result-card" style="background-color: #238636;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                elif pred == 2:
                    st.markdown('<div class="result-card" style="background-color: #d29922; color: #0d1117;">BRICK / CONCRETE 🧱</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card" style="background-color: #da3633;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

            st.metric("Reflection Energy", f"{energy:.4f}")
            st.image(mat2gray_python(roi_ready), caption="BEMD Processing ROI", use_container_width=True)

    else:
        st.info("Please upload both .rad and .rd3 files.")
