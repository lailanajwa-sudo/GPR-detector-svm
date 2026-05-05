import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. KONFIGURASI ---
st.set_page_config(page_title="GPR Object Classifier", layout="wide")

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuatkan fail .pkl: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. FUNGSI PEMPROSESAN ---
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
    # Simulasi IMF1
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    
    # Flatten secara Column-Major ('F')
    feat = imf1_gray.flatten(order='F')
    
    # --- HARD FIX: PASTIKAN 12,000 FEATURES ---
    # Jika feat kurang dari 12000, kita tambah (pad) dengan 0 di hujung
    # Jika feat lebih, kita potong
    if len(feat) < 12000:
        feat = np.pad(feat, (0, 12000 - len(feat)), 'constant')
    else:
        feat = feat[:12000]
        
    return feat.reshape(1, -1)

# --- 3. UI STREAMLIT ---
st.title("📡 GPR BEMD-SVM Detector")

if model is not None:
    st.sidebar.header("Tetapan")
    v_pos = st.sidebar.slider("Vertical", 0, 212, 115)
    h_pos = st.sidebar.slider("Horizontal", 0, 450, 210)
    sens = st.sidebar.slider("Sensitivity", 0.001, 0.050, 0.012, format="%.3f")

    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))

        # Header Processing
        samples = 312
        try:
            content = rad_f.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        except: pass
        
        # Binary Processing
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw) // samples
        
        if num_traces > 0:
            matrix = raw[:samples*num_traces].reshape((samples, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)
            
            # ROI 100x120
            y1, x1 = min(v_pos, samples-100), min(h_pos, num_traces-120)
            roi_crop = full_img[y1:y1+100, x1:x1+120]
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            
            energy = np.std(roi_ready)

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(full_img, cmap='gray', aspect='auto')
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='r', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                if energy < sens:
                    st.info("### Latar Belakang ⚪")
                else:
                    # PROSES PREDICTION
                    features = extract_bemd_features(roi_ready)
                    # Sini ralat tadi berlaku, sekarang dah fix ke 12000
                    scaled_x = scaler.transform(features)
                    pred = model.predict(scaled_x)[0]
                    
                    labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                    res = labels.get(pred, "Unknown")
                    
                    if pred == 1: st.success(f"### {res} ✅")
                    elif pred == 2: st.warning(f"### {res} 🧱")
                    else: st.error(f"### {res} ⚙️")
                    
                    st.write(f"Energy Score: `{energy:.4f}`")
                    st.image(mat2gray_python(roi_ready), caption="Input ROI")
