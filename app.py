import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.signal import detrend

# --- 1. PENGURUSAN ASSET ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Gagal muat model: Pastikan svm_model.pkl & scaler.pkl ada dalam GitHub. ({e})")
        return None, None

model, scaler = load_assets()

# --- 2. FUNGSI PERSIS MATLAB (bemdgpr.m) ---
def mat2gray_python(img):
    """Meniru mat2gray MATLAB: Skala data ke 0.0 - 1.0"""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img)

def extract_matlab_features(roi_matrix):
    """
    Simulasi proses BEMD IMF_1 MATLAB.
    Menggunakan detrending 2D dan Column-Major Flattening (order='F').
    """
    # 1. Detrending untuk simulasi high-pass IMF_1 (seperti fungsi bemd)
    imf1 = detrend(detrend(roi_matrix, axis=0), axis=1)
    
    # 2. MATLAB baris 33: imf1 = mat2gray(IMF_1)
    # Ini memastikan range data adalah 0-1 sebelum masuk ke SVM
    imf1_gray = mat2gray_python(imf1)
    
    # 3. MATLAB baris 41: cavity(:,n) = IMF_1(:)
    # Wajib order='F' (Fortran/MATLAB style) untuk 12,000 features
    return imf1_gray.flatten(order='F').reshape(1, -1)

# --- 3. ANTARAMUKA PENGGUNA (UI) ---
st.set_page_config(layout="wide", page_title="GPR MATLAB-Sync Classifier")
st.title("📡 GPR Precise Target Classifier")

st.sidebar.header("Tetapan ROI (100x120)")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)

# Slider sensitiviti untuk selesaikan masalah 'selalu metal pipe' pada kawasan kosong
energy_threshold = st.sidebar.slider("Threshold Latar Belakang", 0.001, 0.050, 0.012, format="%.3f")

uploaded_files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Baca RAD (Dapatkan SAMPLES)
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # B. Baca RD3 (Data Mentah)
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples
        
        if num_traces > 0:
            # Reshape mengikut Column-Major (order='F')
            matrix = raw_data[:samples*num_traces].reshape((samples, num_traces), order='F')
            
            # --- C. PROSES PENYELARASAN MATLAB ---
            # 1. Background removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 2. MATLAB baris 11: img1 = mat2gray(img)
            matrix_gray = mat2gray_python(matrix_clean)
            
            # D. Potong ROI (100x120)
            y1, x1 = min(v_pos, samples-100), min(h_pos, num_traces-120)
            roi_raw = matrix_gray[y1:y1+100, x1:x1+120]
            
            # E. Semakan Tenaga (Energy Check)
            # Mengelakkan kawasan kosong dikesan sebagai Metal Pipe
            roi_std = np.std(roi_raw)
            
            if roi_std < energy_threshold:
                result_text = "Tiada Objek / Background"
                result_color = "info"
            else:
                # F. Resize & Ekstrak Ciri (Ikut baris 30-33 kod MATLAB)
                img_pil = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
                roi_ready = np.array(img_pil)
                
                # Ekstrak 12,000 features dengan order='F'
                features = extract_matlab_features(roi_ready)
                
                # G. Ramalan SVM
                scaled_input = scaler.transform(features)
                pred_idx = model.predict(scaled_input)[0]
                
                # Pemetaan label (Ikut susunan cvt, con, met dalam MATLAB)
                label_map = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result_text = label_map.get(pred_idx, "Unknown")
                result_color = "success" if pred_idx == 1 else "error"

            # --- H. PAPARAN ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Radargram (mat2gray)")
                fig, ax = plt.subplots()
                ax.imshow(matrix_gray, cmap='gray', aspect='auto')
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                st.subheader("Keputusan Analisis")
                if result_color == "success":
                    st.success(f"### {result_text} ✅")
                elif result_color == "info":
                    st.info(f"### {result_text} ⚪")
                else:
                    st.error(f"### {result_text}")
                
                st.write(f"**Energy ROI:** `{roi_std:.4f}`")
                
                # Paparan tekstur yang dihantar ke SVM
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_raw, cmap='jet')
                ax2.set_title("Input ROI ke BEMD")
                st.pyplot(fig2)
