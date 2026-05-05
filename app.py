import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- SETUP PAGE ---
st.set_page_config(page_title="GPR Object Detector", layout="wide")

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
    # Ikut step MATLAB: BEMD -> IMF1 -> mat2gray -> Flatten
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    feat = imf1_gray.flatten(order='F')
    
    # PAKSA JADI 11,999 (Sama dengan model training)
    return feat[:11999].reshape(1, -1)

# --- UI ---
st.title("📡 GPR BEMD-SVM Classifier")

if model is None:
    st.error("Muat naik fail .pkl baru dari Colab ke GitHub!")
else:
    st.sidebar.header("Tetapan ROI")
    v_pos = st.sidebar.slider("Kedalaman (Y)", 0, 212, 115)
    h_pos = st.sidebar.slider("Trace (X)", 0, 450, 210)
    sens = st.sidebar.slider("Sensitiviti Latar", 0.001, 0.050, 0.015, format="%.3f")

    uploaded = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(uploaded) == 2:
        rad_f = next(f for f in uploaded if f.name.endswith('.rad'))
        rd3_f = next(f for f in uploaded if f.name.endswith('.rd3'))

        # Header RAD
        samples = 312
        content = rad_f.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        
        # Data Binary
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        matrix = raw[:samples*traces].reshape((samples, traces), order='F')
        
        # Process
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
        img_gray = mat2gray_python(matrix_clean)
        
        y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
        roi_crop = img_gray[y1:y1+100, x1:x1+120]
        roi_resized = matlab_resize_manual(roi_crop, (100, 120))
        
        energy = np.std(roi_resized)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots()
            ax.imshow(img_gray, cmap='gray', aspect='auto')
            ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='r', fill=False, lw=2))
            st.pyplot(fig)
            
        with col2:
            if energy < sens:
                st.info("### Keputusan: Latar Belakang ⚪")
            else:
                f = extract_bemd_features(roi_resized)
                s = scaler.transform(f)
                p = model.predict(s)[0]
                
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                res = labels.get(p, "Unknown")
                
                if p == 1: st.success(f"### {res} ✅")
                elif p == 2: st.warning(f"### {res} 🧱")
                else: st.error(f"### {res} ⚙️")
                
                st.write(f"**Energy Score:** `{energy:.4f}`")
                st.image(mat2gray_python(roi_resized), caption="Input ROI")
