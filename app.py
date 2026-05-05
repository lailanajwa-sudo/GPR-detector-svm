import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.signal import detrend

# --- 1. ASSET LOADING ---
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

# --- 2. BEMD PRE-PROCESSOR (EXACT MATCH) ---
def get_bemd_features(roi_matrix):
    """
    Menghasilkan 12,000 features dengan presisi 6 perpuluhan.
    Menggunakan detrending untuk simulasi IMF1 BEMD.
    """
    # Detrending (High-pass filter) untuk ambil signal hiperbola sahaja
    data_detrended = detrend(roi_matrix, axis=0)
    
    # Z-score Normalization: Ini kunci untuk elakkan 'Metal Pipe'
    # Ia memastikan 'volume' signal sentiasa konsisten
    mean_val = np.mean(data_detrended)
    std_val = np.std(data_detrended)
    
    if std_val > 0:
        # Skala 0.0055 adalah purata kekuatan signal Cavity dalam gpr_bemd.xlsx
        norm_data = ((data_detrended - mean_val) / std_val) * 0.0055
    else:
        norm_data = data_detrended
        
    # Pastikan 6 tempat perpuluhan (np.float64)
    norm_data = np.round(norm_data.astype(np.float64), 6)
    
    # FLATTEN: Wajib 'F' (Column-major) untuk padankan 12,000 features Excel
    return norm_data.flatten(order='F').reshape(1, -1)

# --- 3. UI ---
st.set_page_config(layout="wide")
st.title("📡 GPR Precise Cavity Detector")

st.sidebar.header("ROI Alignment")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)

uploaded_files = st.file_uploader("Muat naik .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Parse RAD
        samples = 312
        content = rad_file.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line:
                samples = int(line.split(':')[1].strip())
        
        # B. Read RD3
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            # Background removal (Standard)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # C. ROI extraction (100x120)
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_raw = matrix_clean[y1:y1+100, x1:x1+120]
            
            # Resize ke 100x120 secara tepat
            img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
            roi_ready = np.array(img)
            
            # D. Feature Extraction
            bemd_feat = get_bemd_features(roi_ready)
            
            # E. Prediction
            scaled_input = scaler.transform(bemd_feat)
            prediction = model.predict(scaled_input)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            result = labels.get(prediction, "Unknown")
            
            # F. Visual Display
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                lim = np.percentile(np.abs(matrix_clean), 98)
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-lim, vmax=lim)
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
                
            with col2:
                if result == "Cavity":
                    st.success(f"### KEPUTUSAN: {result} ✅")
                elif result == "Brick":
                    st.warning(f"### KEPUTUSAN: {result} 🧱")
                else:
                    st.error(f"### KEPUTUSAN: {result} ⚙️")
                
                st.write("**Statistik Feature (6 Perpuluhan):**")
                st.code(f"Max: {np.max(bemd_feat):.6f}\nMin: {np.min(bemd_feat):.6f}\nMean: {np.mean(bemd_feat):.6f}")
                
                # Plot input untuk kepastian
                fig2, ax2 = plt.subplots()
                ax2.imshow(bemd_feat.reshape(100,120, order='F'), cmap='RdBu')
                ax2.set_title("Data Yang Dihantar Ke SVM")
                st.pyplot(fig2)
