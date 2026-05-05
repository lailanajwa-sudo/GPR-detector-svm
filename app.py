import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="GPR Object Classifier Pro",
    page_icon="📡",
    layout="wide"
)

# --- 1. MUAT TURUN MODEL & SCALER ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        # Fail ini mesti sepadan dengan 11999 features dari training Colab anda
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. FUNGSI PEMPROSESAN (MATLAB SYNC) ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Meniru interpolasi manual kod MATLAB bemd_gpr.m anda"""
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    """Mengekstrak ciri dan menyelaraskan jumlah features ke 11999"""
    # Detrending 2D (Simulasi IMF1)
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    
    # Flatten secara Column-Major (order='F') - SANGAT PENTING untuk MATLAB sync
    feat = imf1_gray.flatten(order='F')
    
    # FIX: Paksa jumlah features jadi 11,999 (Potong 1 data terakhir dari 12,000)
    # Ini menyelesaikan ralat 'StandardScaler expecting 11999'
    feat_final = feat[:11999].reshape(1, -1)
    return feat_final

# --- 3. ANTARAMUKA PENGGUNA (UI) ---
st.title("📡 GPR BEMD-SVM Target Classifier")
st.markdown("---")

if model is None or scaler is None:
    st.error("⚠️ Fail model/scaler tidak dijumpai! Sila muat naik `svm_model.pkl` dan `scaler.pkl` ke GitHub.")
else:
    # Sidebar untuk kawalan demo
    st.sidebar.header("Tetapan ROI (100x120)")
    v_pos = st.sidebar.slider("Kedalaman ROI (Vertical)", 0, 212, 115)
    h_pos = st.sidebar.slider("Trace ROI (Horizontal)", 0, 450, 210)
    sensitivity = st.sidebar.slider("Sensitiviti Latar", 0.001, 0.050, 0.015, format="%.3f")

    # Upload fail GPR
    uploaded_files = st.file_uploader("Muat naik fail .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(uploaded_files) == 2:
        rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
        rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

        # Baca Header RAD
        samples = 312
        content = rad_file.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line:
                samples = int(line.split(':')[1].strip())
        
        # Baca Data Binary RD3
        raw_data = np.frombuffer(rd3_f_data := rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples
        
        if num_traces > 0:
            # Reshape mengikut Column-Major ('F')
            matrix = raw_data[:samples*num_traces].reshape((samples, num_traces), order='F')
            
            # 1. Background Removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            # 2. Mat2gray
            full_img_gray = mat2gray_python(matrix_clean)
            
            # 3. Potong ROI & Resize Manual
            y1, x1 = min(v_pos, samples-100), min(h_pos, num_traces-120)
            roi_crop = full_img_gray[y1:y1+100, x1:x1+120]
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            
            # Semakan Tenaga (Energy Check)
            energy = np.std(roi_ready)

            # Paparan Keputusan
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Radargram Analisis")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(full_img_gray, cmap='gray', aspect='auto')
                # Tandakan ROI
                rect = patches.Rectangle((x1, y1), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Hasil Klasifikasi")
                if energy < sensitivity:
                    st.info("### Latar Belakang ⚪")
                    st.write("Kawasan ini dikesan sebagai noise atau tanah kosong.")
                else:
                    # Klasifikasi SVM
                    features = extract_bemd_features(roi_ready)
                    scaled_input = scaler.transform(features)
                    prediction = model.predict(scaled_input)[0]
                    
                    # Label Map (Ikut 1:Cavity, 2:Brick, 3:Metal dari Colab)
                    labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                    result = labels.get(prediction, "Unknown")
                    
                    if prediction == 1:
                        st.success(f"### {result} ✅")
                    elif prediction == 2:
                        st.warning(f"### {result} 🧱")
                    else:
                        st.error(f"### {result} ⚙️")
                    
                    st.write(f"**Energy ROI:** `{energy:.4f}`")
                    st.image(mat2gray_python(roi_ready), caption="Input ROI ke SVM", use_column_width=True)
