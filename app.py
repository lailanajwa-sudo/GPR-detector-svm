import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="GPR Object Classifier Pro",
    page_icon="📡",
    layout="wide"
)

# --- 2. MUAT TURUN ASSET (MODEL & SCALER) ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        # Memuatkan model yang dilatih di Colab
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Gagal muat asset: {e}")
        return None, None

model, scaler = load_assets()

# --- 3. FUNGSI PEMPROSESAN DATA ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Replikasi fungsi imresize manual MATLAB"""
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    """Mengekstrak ciri dan menyelaraskan jumlah features ke 11999"""
    # Detrending 2D (Simulasi IMF1 MATLAB)
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    
    # Flatten secara Column-Major ('F') - Sangat penting untuk selari dengan MATLAB
    feat = imf1_gray.flatten(order='F')
    
    # FIX: Paksa jumlah features jadi 11,999 (Potong 1 data terakhir jika 12,000)
    # Ini untuk memadankan input dengan StandardScaler yang mengharap 11,999
    if len(feat) > 11999:
        feat_final = feat[:11999]
    elif len(feat) < 11999:
        # Jika kurang (jarang berlaku), tambah padding 0
        feat_final = np.pad(feat, (0, 11999 - len(feat)), 'constant')
    else:
        feat_final = feat
        
    return feat_final.reshape(1, -1)

# --- 4. ANTARAMUKA PENGGUNA (UI) ---
st.title("📡 GPR BEMD-SVM Target Classifier")
st.markdown("---")

if model is None or scaler is None:
    st.warning("⚠️ Sila pastikan `svm_model.pkl` dan `scaler.pkl` dimuat naik ke GitHub.")
else:
    # Sidebar
    st.sidebar.header("Kawalan ROI")
    v_pos = st.sidebar.slider("Vertical (Depth)", 0, 212, 115)
    h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)
    sensitivity = st.sidebar.slider("Sensitiviti Latar", 0.001, 0.050, 0.012, format="%.3f")

    files = st.file_uploader("Upload fail .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_file = next(f for f in files if f.name.endswith('.rad'))
        rd3_file = next(f for f in files if f.name.endswith('.rd3'))

        # Baca Header RAD
        samples = 312
        try:
            content = rad_file.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line:
                    samples = int(line.split(':')[1].strip())
        except: pass
        
        # Baca Data Binary RD3
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples
        
        if num_traces > 0:
            # Reshape Data
            matrix = raw_data[:samples*num_traces].reshape((samples, num_traces), order='F')
            
            # 1. Background Removal & Normalization
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)
            
            # 2. Ambil ROI & Resize
            y1, x1 = min(v_pos, samples-100), min(h_pos, num_traces-120)
            roi_crop = full_img[y1:y1+100, x1:x1+120]
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            
            # 3. Energy Check
            energy = np.std(roi_ready)

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Radargram")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(full_img, cmap='gray', aspect='auto')
                rect = patches.Rectangle((x1, y1), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Hasil Analisis")
                if energy < sensitivity:
                    st.info("### Latar Belakang ⚪")
                    st.write("Tiada anomali ketara dikesan.")
                else:
                    # Klasifikasi
                    features = extract_bemd_features(roi_ready)
                    scaled_x = scaler.transform(features)
                    pred = model.predict(scaled_x)[0]
                    
                    label_map = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                    result = label_map.get(pred, "Unknown")
                    
                    if pred == 1:
                        st.success(f"### {result} ✅")
                    elif pred == 2:
                        st.warning(f"### {result} 🧱")
                    else:
                        st.error(f"### {result} ⚙️")
                    
                    st.write(f"**Energy Score:** `{energy:.4f}`")
                    st.image(mat2gray_python(roi_ready), caption="Input ROI (mat2gray)")
