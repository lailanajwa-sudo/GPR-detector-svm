import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.ndimage import gaussian_filter # Untuk simulasi BEMD yang laju

# 1. LOAD MODEL & SCALER
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuatkan model: {e}")
        return None, None

model, scaler = load_assets()

# 2. BEMD FEATURE EXTRACTOR (Simulasi IMF1)
def extract_bemd_features(roi_matrix):
    # BEMD IMF1 biasanya bertindak sebagai high-pass filter
    # Kita buang 'trend' frekuensi rendah untuk dapatkan tekstur hiperbola sahaja
    low_freq = gaussian_filter(roi_matrix, sigma=2)
    imf1 = roi_matrix - low_freq
    
    # Normalisasi supaya range sama dengan Excel (sekitar -0.05 ke 0.05)
    # Ini memastikan SVM tidak 'terkejut' dengan nilai besar
    std_excel = 0.0055 # Purata std dev dalam file gpr_bemd.xlsx anda
    current_std = np.std(imf1)
    if current_std > 0:
        imf1_scaled = (imf1 / current_std) * std_target
    else:
        imf1_scaled = imf1
        
    # Flatten menggunakan 'F' (Fortran/MATLAB order) untuk 12,000 features
    return imf1_scaled.flatten(order='F').reshape(1, -1)

# 3. INTERFACE UTAMA
st.set_page_config(layout="wide", page_title="GPR BEMD-SVM")
st.title("📡 GPR BEMD Feature Classifier")

# Sidebar untuk ROI
st.sidebar.header("Tetapan ROI (100x120)")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)
std_target = st.sidebar.slider("Sensitiviti (Std Dev)", 0.001, 0.020, 0.0055, format="%.4f")

uploaded_files = st.file_uploader("Muat naik .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Baca RAD
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # B. Baca RD3 (Double Precision)
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            # Background Removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # C. Extract ROI 100x120
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_raw = matrix_clean[y1:y1+100, x1:x1+120]
            
            # Resize (Gunakan PIL untuk kualiti tinggi tanpa cv2)
            img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
            roi_resized = np.array(img)
            
            # D. PROSES BEMD FEATURES
            # Kita tukar 100x120 pixel jadi 12,000 ciri BEMD
            bemd_features = extract_bemd_features(roi_resized)
            
            # E. PREDICTION
            scaled_input = scaler.transform(bemd_features)
            pred = model.predict(scaled_input)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            result = labels.get(pred, "Unknown")
            
            # F. DISPLAY
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Visualisasi Radargram")
                fig, ax = plt.subplots()
                lim = np.percentile(np.abs(matrix_clean), 98)
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-lim, vmax=lim)
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                st.header("Hasil Klasifikasi")
                if result == "Cavity":
                    st.success(f"### {result} ✅")
                elif result == "Brick":
                    st.warning(f"### {result} 🧱")
                else:
                    st.error(f"### {result} ⚙️")
                
                # Semakan Ketepatan (6 Perpuluhan)
                st.write("**Statistik Ciri BEMD:**")
                st.code(f"Max Val: {np.max(bemd_features):.6f}\nMin Val: {np.min(bemd_features):.6f}\nMean: {np.mean(bemd_features):.6f}")
                
                # Preview apa yang SVM nampak
                fig2, ax2 = plt.subplots()
                ax2.imshow(bemd_features.reshape(100, 120, order='F'), cmap='jet')
                ax2.set_title("Ciri BEMD (Input ke SVM)")
                st.pyplot(fig2)
        else:
            st.error("Data RD3 tidak sah.")
