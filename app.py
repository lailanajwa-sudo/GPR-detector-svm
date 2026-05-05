import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.signal import detrend

# --- 1. LOAD ASSETS ---
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

# --- 2. MATLAB REPLICATION FUNCTIONS ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    # Wajib order='F' (Column-Major) untuk padankan MATLAB IMF_1(:)
    return imf1_gray.flatten(order='F').reshape(1, -1)

# --- 3. UI STREAMLIT ---
st.set_page_config(layout="wide")
st.title("📡 GPR Precise Target Classifier")

if model is None:
    st.error("Sila muat naik fail .pkl yang baru dihasilkan dari Colab ke GitHub.")
else:
    st.sidebar.header("Tetapan ROI")
    v_pos = st.sidebar.slider("Vertical (Depth)", 0, 212, 115)
    h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)
    threshold = st.sidebar.slider("Sensitiviti Latar", 0.001, 0.050, 0.015, format="%.3f")

    uploaded = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(uploaded) == 2:
        rad = next(f for f in uploaded if f.name.endswith('.rad'))
        rd3 = next(f for f in uploaded if f.name.endswith('.rd3'))

        # Header
        samples = 312
        content = rad.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        
        # Binary Data
        raw = np.frombuffer(rd3.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            
            # MATLAB Flow
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            img1 = mat2gray_python(matrix_clean)
            
            # ROI Processing
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            img2 = img1[y1:y1+100, x1:x1+120]
            img3 = matlab_resize_manual(img2, (100, 120))
            
            energy = np.std(img3)
            if energy < threshold:
                st.info("### Keputusan: Latar Belakang / Kosong ⚪")
            else:
                features = extract_bemd_features(img3)
                scaled_x = scaler.transform(features)
                pred = model.predict(scaled_x)[0]
                
                label_map = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                res = label_map.get(pred, "Unknown")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(img1, cmap='gray', aspect='auto')
                    ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                    st.pyplot(fig)
                with col2:
                    if pred == 1: st.success(f"### {res} ✅")
                    elif pred == 2: st.warning(f"### {res} 🧱")
                    else: st.error(f"### {res} ⚙️")
                    st.write(f"**Energy Score:** `{energy:.4f}`")
                    st.image(mat2gray_python(img3), caption="ROI Zoomed (Input SVM)")
