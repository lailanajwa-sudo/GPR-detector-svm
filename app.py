import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.signal import detrend

# --- 1. MUAT TURUN MODEL ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Gagal muat model: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. FUNGSI PERSIS MATLAB ---
def mat2gray_python(img):
    """Meniru mat2gray MATLAB"""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img)

def extract_features(roi_matrix):
    """Simulasi proses BEMD IMF_1 MATLAB"""
    # Detrending untuk simulasi high-pass IMF_1
    imf1 = detrend(detrend(roi_matrix, axis=0), axis=1)
    # PENTING: mat2gray pada IMF_1 seperti baris 33 kod MATLAB anda
    imf1_gray = mat2gray_python(imf1)
    # Flattening Column-Major (order='F') untuk padankan IMF_1(:)
    return imf1_gray.flatten(order='F').reshape(1, -1)

# --- 3. ANTARAMUKA PENGGUNA ---
st.set_page_config(layout="wide")
st.title("📡 GPR Precise Classifier")

st.sidebar.header("Konfigurasi ROI")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)
# Nilai threshold untuk abaikan kawasan kosong
sensitivity = st.sidebar.slider("Threshold Sensitiviti", 0.001, 0.050, 0.015, format="%.3f")

uploaded_files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Baca RAD
        samples = 312
        content = rad_file.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line:
                samples = int(line.split(':')[1].strip())
        
        # B. Baca RD3
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            
            # C. Pre-processing MATLAB (Baris 11)
            # Background removal + mat2gray
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            matrix_gray = mat2gray_python(matrix_clean)
            
            # D. ROI Extraction (100x120)
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_raw = matrix_gray[y1:y1+100, x1:x1+120]
            
            # E. Semakan Tenaga (Cegah salah kesan kawasan kosong)
            roi_std = np.std(roi_raw)
            
            if roi_std < sensitivity:
                result = "Tiada Objek (Background)"
                status = "info"
            else:
                # F. Resize & Feature Extraction (Baris 30-33)
                img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
                roi_ready = np.array(img)
                features = extract_features(roi_ready)
                
                # G. Prediction
                scaled_input = scaler.transform(features)
                pred = model.predict(scaled_input)[0]
                label_map = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = label_map.get(pred, "Tidak Diketahui")
                status = "success" if pred == 1 else "error"

            # H. Paparan
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(matrix_gray, cmap='gray', aspect='auto')
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                st.subheader("Hasil Analisis")
                if status == "success": st.success(f"### {result} ✅")
                elif status == "info": st.info(f"### {result} ⚪")
                else: st.error(f"### {result}")
                
                st.write(f"**Tenaga ROI:** `{roi_std:.4f}`")
                st.write(f"**Saiz Features:** `{features.shape}`")
