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
    """Memuatkan model SVM dan Scaler dengan laluan yang teguh."""
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Ralat memuatkan fail model: Pastikan svm_model.pkl dan scaler.pkl berada dalam repository. ({e})")
        return None, None

model, scaler = load_assets()

# --- 2. LOGIK PEMPROSESAN (MENIRU MATLAB BEMD_GPR.M) ---
def mat2gray_python(img):
    """Meniru fungsi mat2gray MATLAB untuk menukar range data ke 0.0 - 1.0."""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img)

def get_bemd_features(roi_matrix):
    """
    Simulasi IMF_1 dari BEMD MATLAB.
    Menggunakan 2D Detrending dan Column-Major Flattening (order='F').
    """
    # 1. Detrending untuk simulasi high-pass IMF_1
    imf1_sim = detrend(detrend(roi_matrix, axis=0), axis=1)
    
    # 2. MATLAB: imf1 = mat2gray(IMF_1) - Berdasarkan baris 33 dalam kod MATLAB anda
    imf1_gray = mat2gray_python(imf1_sim)
    
    # 3. MATLAB: cavity(:,n) = IMF_1(:) -> Wajib order='F' untuk 12,000 features
    return imf1_gray.flatten(order='F').reshape(1, -1)

# --- 3. ANTARAMUKA PENGGUNA (UI) ---
st.set_page_config(layout="wide", page_title="GPR BEMD-SVM Detector")
st.title("📡 GPR Target Classifier (MATLAB Synced)")

# Sidebar untuk kawalan ROI
st.sidebar.header("Pelarasan Kotak Merah (ROI)")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)

# Threshold untuk mengesan kawasan kosong
energy_threshold = st.sidebar.slider("Threshold Sensitiviti", 0.001, 0.050, 0.015, format="%.3f")

uploaded_files = st.file_uploader("Muat naik fail .rad dan .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Baca fail RAD untuk mendapatkan jumlah SAMPLES
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # B. Baca fail RD3 (Data mentah)
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples
        
        if num_traces > 0:
            # Reshape mengikut Column-Major (MALA standard)
            matrix = raw_data[:samples*num_traces].reshape((samples, num_traces), order='F')
            
            # --- C. PEMPROSESAN DATA ---
            # 1. Background removal (Sama seperti readmala)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 2. MATLAB baris 11: img1 = mat2gray(img)
            matrix_gray = mat2gray_python(matrix_clean)
            
            # D. Ekstrak ROI (Saiz tetap 100x120 seperti dalam Excel)
            y1, x1 = min(v_pos, samples-100), min(h_pos, num_traces-120)
            roi_raw = matrix_gray[y1:y1+100, x1:x1+120]
            
            # E. SEMAKAN TENAGA (Elak kawasan kosong dikesan sebagai Metal)
            energy = np.std(roi_raw)
            
            if energy < energy_threshold:
                result_text = "Background / No Pattern"
                result_color = "info"
            else:
                # F. RESIZE & EXTRACTION
                # Resize menggunakan BICUBIC supaya kualiti sama dengan MATLAB imresize
                img_pil = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
                roi_ready = np.array(img_pil)
                
                # Ekstrak 12,000 ciri BEMD
                features = get_bemd_features(roi_ready)
                
                # G. RAMALAN SVM
                scaled_input = scaler.transform(features)
                pred_idx = model.predict(scaled_input)[0]
                
                label_map = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result_text = label_map.get(pred_idx, "Unknown")
                result_color = "success" if pred_idx == 1 else "error"

            # --- H. PAPARAN KEPUTUSAN ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Visualisasi Radargram")
                fig, ax = plt.subplots()
                # Paparan menggunakan grayscale
                ax.imshow(matrix_gray, cmap='gray', aspect='auto')
                # Lukis kotak merah ROI
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                st.subheader("Hasil Analisis")
                if result_color == "success":
                    st.success(f"### {result_text} ✅")
                elif result_color == "info":
                    st.info(f"### {result_text} ⚪")
                else:
                    st.error(f"### {result_text}")
                
                st.write(f"**Tahap Tenaga ROI:** `{energy:.4f}`")
                
                # Preview apa yang SVM nampak
                st.write("**Data Input SVM (12k features):**")
                fig2, ax2 = plt.subplots()
                # Reshape semula untuk paparan visual sahaja
                ax2.imshow(roi_raw, cmap='jet')
                ax2.set_title("Tekstur ROI")
                st.pyplot(fig2)
        else:
            st.warning("Data RD3 tidak mencukupi atau rosak.")
