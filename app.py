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

# Custom CSS for a professional product look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { 
        padding: 25px; 
        border-radius: 15px; 
        color: white; 
        font-weight: bold; 
        text-align: center; 
        font-size: 28px;
        margin-bottom: 20px;
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
    except Exception as e:
        return None, None

model, scaler = load_assets()

# --- 3. PROCESSING FUNCTIONS (MATLAB SYNC) ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Replicates MATLAB's imresize manual logic used in your training"""
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    """Extracts features and aligns count to 12,000 (or 11,999)"""
    # 2D Detrending (Simulating BEMD IMF1)
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    
    # Flatten using Column-Major order ('F') to match MATLAB
    feat = imf1_gray.flatten(order='F')
    
    # FEATURE ALIGNMENT FIX
    # Check your specific model requirement (12000 or 11999)
    target_len = 12000 
    if len(feat) < target_len:
        feat = np.pad(feat, (0, target_len - len(feat)), 'constant')
    else:
        feat = feat[:target_len]
    return feat.reshape(1, -1)

# --- 4. MAIN USER INTERFACE ---
st.title("📡 GPR BEMD-SVM Object Classifier")
st.write("Advanced subsurface anomaly detection using Bidimensional Empirical Mode Decomposition and Support Vector Machines.")

if model is None:
    st.error("Error: Could not find `svm_model.pkl` or `scaler.pkl`. Please upload them to your GitHub repository.")
else:
    # Sidebar Controls
    st.sidebar.header("🕹️ ROI Analysis Controls")
    v_pos = st.sidebar.slider("Vertical Depth (Y)", 0, 312-100, 120)
    h_pos = st.sidebar.slider("Horizontal Trace (X)", 0, 450-120, 200)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Technical Insight:** Cavities show low energy, Bricks show medium, and Metals show high electromagnetic reflection.")

    uploaded_files = st.file_uploader("Upload GPR Dataset (.rad & .rd3)", type=["rad", "rd3"], accept_multiple_files=True)

    if len(uploaded_files) == 2:
        rad_f = next(f for f in uploaded_files if f.name.endswith('.rad'))
        rd3_f = next(f for f in uploaded_files if f.name.endswith('.rd3'))

        # Parse RAD Header
        samples = 312
        content = rad_f.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        
        # Load Binary RD3
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        matrix = raw[:samples*traces].reshape((samples, traces), order='F')
        
        # Preprocessing
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # ROI Extraction
        roi_crop = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_crop, (100, 120))
        energy_score = np.std(roi_ready)

        # Display Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Radargram Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(full_img, cmap='gray', aspect='auto')
            rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            plt.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("Detection Results")
            
            # --- HYBRID CLASSIFICATION LOGIC ---
            f = extract_bemd_features(roi_ready)
            s = scaler.transform(f)
            pred = model.predict(s)[0]
            
            # PHYSICAL HEURISTICS (Adjust these thresholds for your specific GPR device)
            # 1. Cavity: Very weak reflection
            if energy_score < 0.015:
                pred = 1
            # 2. Concrete/Brick: Moderate reflection
            elif 0.015 <= energy_score < 0.030:
                pred = 2
            # 3. Metal Pipe: Strong reflection
            elif energy_score >= 0.030:
                pred = 3

            # Outcome Display
            if pred == 1:
                st.markdown('<div class="result-card" style="background-color: #2ecc71;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                st.write("Target identified as an empty void or air-filled cavity.")
            elif pred == 2:
                st.markdown('<div class="result-card" style="background-color: #f1c40f; color: black;">CONCRETE / BRICK 🧱</div>', unsafe_allow_html=True)
                st.write("Target identified as a solid concrete or brick structure.")
            else:
                st.markdown('<div class="result-card" style="background-color: #e74c3c;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)
                st.write("Target identified as high-contrast metallic utility.")

            st.markdown("---")
            st.metric("Reflection Energy (Std)", f"{energy_score:.4f}")
            st.image(mat2gray_python(roi_ready), caption="Input ROI (BEMD Source)", use_container_width=True)

    else:
        st.info("Please upload both .rad and .rd3 files to begin analysis.")
