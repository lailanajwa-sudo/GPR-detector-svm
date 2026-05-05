import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="GPR Object Detector Pro", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { 
        padding: 30px; border-radius: 15px; color: white; font-weight: bold; 
        text-align: center; font-size: 32px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-label { font-size: 18px; font-weight: bold; color: #34495e; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# --- 3. CORE PROCESSING ENGINE ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    denominator = mx - mn
    if denominator <= 1e-7: # Use epsilon to avoid float issues
        return np.zeros_like(img, dtype=np.float64)
    return (img - mn) / denominator

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    if old_h == 0 or old_w == 0: return np.zeros(new_shape)
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    feat = imf1_gray.flatten(order='F')
    # Use 12000 as standard; adjust to 11999 only if your scaler specifically errors
    target_len = 12000 
    if len(feat) < target_len: feat = np.pad(feat, (0, target_len - len(feat)), 'constant')
    else: feat = feat[:target_len]
    return feat.reshape(1, -1)

# --- 4. MAIN INTERFACE ---
st.title("📡 GPR Intelligent Subsurface Classifier")

if model is None:
    st.error("Missing Assets! Upload `svm_model.pkl` and `scaler.pkl` to GitHub.")
else:
    # Sidebar
    st.sidebar.header("🕹️ Target Navigation")
    v_pos = st.sidebar.slider("Depth (Vertical)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450-125, 200)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sensitivity Tuning")
    # If background still detects as cavity, slide this to the RIGHT (higher)
    soil_limit = st.sidebar.slider("Background Noise Filter", 0.001, 0.025, 0.012, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3 Files", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))

        samples = 312
        try:
            content = rad_f.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        except: pass
        
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)
            
            # Safe ROI Slicing
            r_v, r_h = min(v_pos + 100, samples), min(h_pos + 120, traces)
            roi_crop = full_img[v_pos:r_v, h_pos:r_h]
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            
            # Use energy as a standard indicator
            energy = np.std(roi_ready)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Radargram View")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(full_img, cmap='gray', aspect='auto')
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='#e74c3c', facecolor='none')
                ax.add_patch(rect)
                plt.axis('off')
                st.pyplot(fig)

            with col2:
                st.subheader("Target Analysis")
                
                # --- LOGIC: BACKGROUND VS TARGET ---
                if energy < soil_limit:
                    st.markdown('<div class="result-card" style="background-color: #95a5a6;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                    st.write("Status: Scanning background soil (Noise level: Low)")
                else:
                    f = extract_bemd_features(roi_ready)
                    s = scaler.transform(f)
                    pred = model.predict(s)[0]
                    
                    # Refinement based on Reflection Energy (Physics Logic)
                    if energy < 0.016: pred = 1 # Cavity
                    elif 0.016 <= energy < 0.032: pred = 2 # Concrete/Brick
                    elif energy >= 0.032: pred = 3 # Metal Pipe

                    if pred == 1:
                        st.markdown('<div class="result-card" style="background-color: #2ecc71;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                    elif pred == 2:
                        st.markdown('<div class="result-card" style="background-color: #f1c40f; color: #2c3e50;">BRICK / CONCRETE 🧱</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-card" style="background-color: #e74c3c;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

                st.metric("Reflection Energy (Intensity)", f"{energy:.4f}")
                
                # Visual check for the user
                st.markdown('<p class="status-label">BEMD ROI Input:</p>', unsafe_allow_html=True)
                # Show the processed ROI but only if there is actual signal
                display_roi = mat2gray_python(roi_ready) if energy >= 0.001 else np.zeros((100,120))
                st.image(display_roi, use_container_width=True)
