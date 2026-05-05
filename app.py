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

# --- 2. FUNGSI PERSIS MATLAB (mat2gray & BEMD Feature) ---
def mat2gray_python(img):
    """Meniru fungsi mat2gray MATLAB"""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return img

def get_bemd_features(roi_matrix):
    # MATLAB anda ambil IMF_1. Secara matematik, IMF_1 dalam BEMD 
    # berfungsi sebagai filter residu frekuensi tinggi.
    # Kita gunakan detrending 2D untuk simulasi IMF_1 yang paling dekat.
    imf1_sim = detrend(detrend(roi_matrix, axis=0), axis=1)
    
    # MATLAB: IMF_1(:) -> Ini adalah Column-Major flattening
    # Kita guna order='F' (Fortran/MATLAB order)
    return imf1_sim.flatten(order='F').reshape(1, -1)

# --- 3. UI UTAMA ---
st.set_page_config(layout="wide", page_title="GPR BEMD-SVM Precise")
st.title("📡 GPR Classifier (MATLAB Logic Synced)")

st.sidebar.header("ROI Alignment")
v_pos = st.sidebar.slider("Vertical (Depth)", 0, 212, 115)
h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)

uploaded_files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Parse Header
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # B. Read RD3
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            # Bentuk matriks (Wajib 'F' untuk MALA data)
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            
            # --- C. PROSES PERSIS MATLAB ---
            # 1. Background removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            # 2. mat2gray (Penting: Ikut kod MATLAB anda)
            matrix_gray = mat2gray_python(matrix_clean)
            
            # D. ROI Extraction (100x120)
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_raw = matrix_gray[y1:y1+100, x1:x1+120]
            
            # Resize ke 100x120 (Sama seperti imresize dalam MATLAB anda)
            img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
            roi_ready = np.array(img)
            
            # E. Extract Features (Simulasi IMF_1 + Flattening order='F')
            bemd_feat = get_bemd_features(roi_ready)
            
            # F. SVM Prediction
            scaled_input = scaler.transform(bemd_feat)
            prediction = model.predict(scaled_input)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            result = labels.get(prediction, "Unknown")
            
            # G. Visual
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(matrix_gray, cmap='gray', aspect='auto')
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                if result == "Cavity":
                    st.success(f"### HASIL: {result} ✅")
                elif result == "Brick":
                    st.warning(f"### HASIL: {result} 🧱")
                else:
                    st.error(f"### HASIL: {result} ⚙️")
                
                st.write("**Debug Info:**")
                st.write(f"Shape Features: {bemd_feat.shape}") # Mesti (1, 12000)
                
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_ready, cmap='jet')
                ax2.set_title("Input ROI (Setelah mat2gray)")
                st.pyplot(fig2)
