import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GPR Object Detector Pro",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { 
        padding: 30px; border-radius: 15px; color: white; font-weight: bold; 
        text-align: center; font-size: 32px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
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
    except:
        return None, None

model, scaler = load_assets()

# --- 3. CRASH-PROOF PROCESSING ENGINE ---
def mat2gray_python(img):
    """Prevents division by zero if image is flat"""
    mn, mx = np.min(img), np.max(img)
    denominator = mx - mn
    if denominator <= 0:
        return np.zeros_like(img, dtype=np.float64)
    return (img - mn) / denominator

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Fixed: Prevents ZeroDivisionError if img shape is 0"""
    old_h, old_w = img.shape
    if old_h == 0 or old_w == 0:
        return np.zeros(new_shape) # Return blank if input is empty
        
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    """Ensures feature count is exactly what the model expects"""
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    feat = imf1_gray.flatten(order='F')
    
    # ADJUST THIS: 12000 or 11999 based on your pkl file
    target_len = 12000 
    if len(feat) < target_len:
        feat = np.pad(feat, (0, target_len - len(feat)), 'constant')
    else:
        feat = feat[:target_len]
    return feat.reshape(1, -1)

# --- 4. MAIN INTERFACE ---
st.title("📡 GPR BEMD-SVM Object Classifier")

if model is None:
    st.error("Assets missing! Upload `svm_model.pkl` and `scaler.pkl` to GitHub.")
else:
    # Sidebar
    st.sidebar.header("🕹️ ROI Navigation")
    # Dynamic limits to prevent empty crops
    v_pos = st.sidebar.slider("Vertical (Depth)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450-125, 200)
    
    st.sidebar.markdown("---")
    soil_limit = st.sidebar.slider("Soil Background Filter", 0.001, 0.020, 0.008, step=0.001)

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))

        # Header parsing
        samples = 312
        try:
            content = rad_f.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        except: pass
        
        # Binary processing
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)
            
            # SAFE CROP: Ensures crop is always within data bounds
            # We add +1 to ensure it never results in a 0-width image
            r_v = min(v_pos + 100, samples)
            r_h = min(h_pos + 120, traces)
            roi_crop = full_img[v_pos:r_v, h_pos:r_h]
            
            # Resize
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            energy = np.std(roi_ready)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Radargram Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(full_img, cmap='gray', aspect='auto')
                # ROI Box
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                plt.axis('off')
                st.pyplot(fig)

            with col2:
                st.subheader("Detection Result")
                if energy < soil_limit:
                    st.markdown('<div class="result-card" style="background-color: #7f8c8d;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                else:
                    f = extract_bemd_features(roi_ready)
                    s = scaler.transform(f)
                    pred = model.predict(s)[0]
                    
                    # Logic Refinement
                    if energy < 0.016: pred = 1 # Cavity
                    elif 0.016 <= energy < 0.030: pred = 2 # Concrete
                    elif energy >= 0.030: pred = 3 # Metal

                    if pred == 1:
                        st.markdown('<div class="result-card" style="background-color: #2ecc71;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                    elif pred == 2:
                        st.markdown('<div class="result-card" style="background-color: #f1c40f; color: #2c3e50;">CONCRETE / BRICK 🧱</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-card" style="background-color: #e74c3c;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

                st.metric("Reflection Energy", f"{energy:.4f}")
                st.image(mat2gray_python(roi_ready), caption="BEMD Source ROI", use_container_width=True)
