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

# --- 2. FUNGSI MATLAB (mat2gray & BEMD Feature) ---
def mat2gray_python(img):
    """Meniru fungsi mat2gray MATLAB"""
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return np.zeros_like(img)

def get_bemd_features(roi_matrix):
    # MATLAB anda melakukan detrending melalui fungsi bemd()
    # Kita simulasikan IMF1 dengan membuang trend linear pada ROI
    imf1_sim = detrend(detrend(roi_matrix, axis=0), axis=1)
    
    # PENTING: Kod MATLAB anda menggunakan mat2gray sekali lagi pada IMF_1
    # Ini memastikan semua data (Cavity/Metal) berada pada skala 0-1
    imf1_gray = mat2gray_python(imf1_sim)
    
    # MATLAB: IMF_1(:) -> Column-Major flattening
    return imf1_gray.flatten(order='F').reshape(1, -1)

# --- 3. ANTARAMUKA PENGGUNA ---
st.set_page_config(layout="wide")
st.title("📡 GPR BEMD-SVM (MATLAB Synced)")

st.sidebar.header("ROI Alignment")
v_pos = st.sidebar.slider("Kedalaman (Vertical)", 0, 212, 115)
h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450, 210)

uploaded_files = st.file_uploader("Muat naik .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Baca Header
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # B. Baca RD3
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            
            # --- C. PROSES PENYELARASAN MATLAB ---
            # 1. Background removal (Sama seperti readmala)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            # 2. mat2gray pertama (Baris 11 & 49 & 89 dalam kod MATLAB anda)
            matrix_gray = mat2gray_python(matrix_clean)
            
            # D. Potong ROI (100x120)
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_raw = matrix_gray[y1:y1+100, x1:x1+120]
            
            # E. Resize (Sama seperti imresize dalam MATLAB anda)
            img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
            roi_ready = np.array(img)
            
            # F. Ekstrak Ciri (Simulasi IMF_1 + mat2gray + Flattening order='F')
            bemd_feat = get_bemd_features(roi_ready)
            
            # G. Ramalan SVM
            scaled_input = scaler.transform(bemd_feat)
            prediction = model.predict(scaled_input)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            result = labels.get(prediction, "Unknown")
            
            # H. Paparan Visual
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(matrix_gray, cmap='gray', aspect='auto')
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                if result == "Cavity":
                    st.success(f"### KEPUTUSAN: {result} ✅")
                elif result == "Brick":
                    st.warning(f"### KEPUTUSAN: {result} 🧱")
                else:
                    st.error(f"### KEPUTUSAN: {result} ⚙️")
                
                # Plot feature untuk pengesahan visual
                st.write("**Visual Ciri BEMD (Input SVM):**")
                fig2, ax2 = plt.subplots()
                # Kita reshape semula untuk tengok tekstur yang dihantar ke SVM
                ax2.imshow(bemd_feat.reshape(100, 120, order='F'), cmap='jet')
                st.pyplot(fig2)
