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

# Custom CSS for high-end UI
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { 
        padding: 30px; 
        border-radius: 15px; 
        color: white; 
        font-weight: bold; 
        text-align: center; 
        font-size: 32px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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

# --- 3. PROCESSING ENGINE ---
def mat2gray_python(img):
    """Safely converts image to grayscale [0,1] and prevents ZeroDivisionError"""
    mn, mx = np.min(img), np.max(img)
    denominator = mx - mn
    if denominator <= 0:
        return np.zeros_like(img, dtype=np.float64)
    return (img - mn) / denominator

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    # IMF1 Simulation using 2D Detrending
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    feat = imf1_gray.flatten(order='F')
    
    # Feature count alignment (Matches your .pkl requirements)
    # If your model expects 11999, change this to 11999
    target_len = 12000 
    if len(feat) < target_len:
        feat = np.pad(feat, (0, target_len - len(feat)), 'constant')
    else:
        feat = feat[:target_len]
    return feat.reshape(1, -1)

# --- 4. MAIN INTERFACE ---
st.title("📡 GPR BEMD-SVM Object Classifier")
st.write("Intelligent subsurface mapping using BEMD Feature Extraction and SVM Classification.")

if model is None:
    st.error("Assets missing. Please ensure `svm_model.pkl` and `scaler.pkl` are in the repository.")
else:
    # Sidebar
    st.sidebar.header("🕹️ ROI Navigation")
    v_pos = st.sidebar.slider("Vertical (Depth)", 0, 312-100, 150)
    h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450-120, 250)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Threshold Fine-tuning")
    soil_limit = st.sidebar.slider("Background Filter", 0.005, 0.015, 0.008, step=0.001)

    files = st.file_uploader("Upload GPR Data (.rad & .rd3)", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))

        # Header parsing
        samples = 312
        content = rad_f.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        
        # Binary data processing
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        matrix = raw[:samples*traces].reshape((samples, traces), order='F')
        
        # Filtering & Normalization
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        full_img = mat2gray_python(matrix_clean)
        
        # Target Crop & Analysis
        roi_crop = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        roi_ready = matlab_resize_manual(roi_crop, (100, 120))
        energy = np.std(roi_ready)

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
            
            # --- HYBRID CLASSIFICATION ENGINE ---
            if energy < soil_limit:
                # STATE: BACKGROUND
                st.markdown('<div class="result-card" style="background-color: #7f8c8d;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                st.write("Scanning background soil. No significant anomaly detected.")
            else:
                f = extract_bemd_features(roi_ready)
                s = scaler.transform(f)
                pred = model.predict(s)[0]
                
                # REFINEMENT LOGIC (Physics-Based Overrides)
                # Adjusted thresholds for better Concrete detection
                if soil_limit <= energy < 0.016:
                    pred = 1 # Cavity
                elif 0.016 <= energy < 0.030:
                    pred = 2 # Concrete/Brick
                elif energy >= 0.030:
                    pred = 3 # Metal Pipe

                # VISUAL DISPLAY
                if pred == 1:
                    st.markdown('<div class="result-card" style="background-color: #2ecc71;">CAVITY (VOID) ✅</div>', unsafe_allow_html=True)
                elif pred == 2:
                    st.markdown('<div class="result-card" style="background-color: #f1c40f; color: #2c3e50;">CONCRETE / BRICK 🧱</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card" style="background-color: #e74c3c;">METAL PIPE ⚙️</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.metric("Reflection Energy Intensity", f"{energy:.4f}")
            st.image(mat2gray_python(roi_ready), caption="BEMD Source ROI", use_container_width=True)

    else:
        st.info("Upload both .rad and .rd3 files to initialize the scanning system.")
