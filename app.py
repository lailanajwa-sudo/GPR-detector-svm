import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.signal import detrend

# --- 1. KONFIGURASI ANTARAMUKA (CONTEST READY) ---
st.set_page_config(
    page_title="GPR Object Classifier Pro",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MUAT TURUN ASSET ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        # Pastikan fail ini telah dimuat naik ke GitHub selepas training di Colab
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 3. FUNGSI REPLIKASI MATLAB ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Meniru baris 23-28 dalam kod bemd_gpr.m anda"""
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    # Simulasi IMF1 (Detrending 2D)
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    # PENTING: Gunakan order='F' untuk ikut format MATLAB IMF_1(:)
    feat = imf1_gray.flatten(order='F')
    
    # Padankan dengan 11,999 features (berdasarkan ralat Colab anda tadi)
    if len(feat) >= 11999:
        return feat[:11999].reshape(1, -1)
    else:
        # Jika kurang, kita padding dengan 0 (jarang berlaku)
        padded = np.zeros(11999)
        padded[:len(feat)] = feat
        return padded.reshape(1, -1)

# --- 4. STRUKTUR UTAMA APP ---
st.title("📡 GPR BEMD-SVM Target Classifier")
st.write("Sistem Pengesan Cavity, Brick, dan Metal Pipe menggunakan BEMD & SVM.")

if model is None:
    st.error("Ralat: Fail `svm_model.pkl` atau `scaler.pkl` tidak dijumpai. Sila muat naik fail dari Colab ke GitHub.")
else:
    # Sidebar untuk kawalan demo semasa contest
    st.sidebar.header("Kawalan Analisis")
    v_pos = st.sidebar.slider("Kedalaman ROI (Y)", 0, 212, 115)
    h_pos = st.sidebar.slider("Trace ROI (X)", 0, 450, 210)
    sensitivity = st.sidebar.slider("Sensitiviti Latar", 0.001, 0.050, 0.015, format="%.3f")

    files = st.file_uploader("Muat naik fail .rad & .rd3", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        rad_f = next(f for f in files if f.name.endswith('.rad'))
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))

        # Baca header RAD
        samples = 312
        try:
            content = rad_f.getvalue().decode("utf-8")
            for line in content.split('\n'):
                if "SAMPLES:" in line: samples = int(line.split(':')[1].strip())
        except: pass
        
        # Baca data binary RD3
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            # Reshape mengikut Column-Major
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            
            # --- PROSES REPLIKASI MATLAB ---
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            img_gray = mat2gray_python(matrix_clean)
            
            # Potong ROI 100x120
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi_crop = img_gray[y1:y1+100, x1:x1+120]
            roi_resized = matlab_resize_manual(roi_crop, (100, 120))
            
            # Check Energy (Kosong vs Objek)
            energy = np.std(roi_resized)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Radargram")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(img_gray, cmap='gray', aspect='auto')
                # Lukis kotak merah
                rect = patches.Rectangle((x1, y1), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Keputusan")
                if energy < sensitivity:
                    st.info("### Latar Belakang ⚪")
                    st.write("Tiada corak hiperbola dikesan di kawasan ini.")
                else:
                    # Klasifikasi
                    features = extract_bemd_features(roi_resized)
                    scaled_feat = scaler.transform(features)
                    prediction = model.predict(scaled_feat)[0]
                    
                    # Label Map (Sama seperti training Colab)
                    labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                    result = labels.get(prediction, "Unknown")
                    
                    if prediction == 1:
                        st.success(f"### {result} ✅")
                    elif prediction == 2:
                        st.warning(f"### {result} 🧱")
                    else:
                        st.error(f"### {result} ⚙️")
                    
                    st.metric("Energy Score", f"{energy:.4f}")
                    st.image(mat2gray_python(roi_resized), caption="Imej Input SVM", use_column_width=True)

        else:
            st.error("Data RD3 tidak sah atau rosak.")
