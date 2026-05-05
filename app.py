import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. HALAMAN KONFIGURASI ---
st.set_page_config(
    page_title="GPR Intelligent Subsurface Classifier",
    page_icon="📡",
    layout="wide"
)

# Custom CSS untuk UI premium mod gelap
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    /* Membetulkan kotak putih metrik supaya gelap */
    .stMetric { 
        background-color: #1a1c24; 
        color: white; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); 
    }
    [data-testid="stMetricValue"] { color: lime !important; font-size: 24px; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }

    /* Kotak keputusan yang lebih kemas & teks tidak terpotong */
    .result-card { 
        padding: 25px; 
        border-radius: 15px; 
        color: white; 
        font-weight: bold; 
        text-align: center; 
        font-size: 30px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        line-height: 1.3;
    }
    .noise-card { 
        padding: 20px; 
        border-radius: 10px; 
        color: #aaaaaa; 
        background-color: #1a1c24; 
        text-align: center; 
        font-size: 18px; 
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MUAT TURUN MODEL & SCALER ---
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

# --- 3. FUNGSI PEMPROSESAN DATA (MATLAB SYNC) ---
def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    denominator = mx - mn
    # Menggunakan epsilon kecil untuk mengelakkan ralat ZeroDivision
    if denominator <= 1e-9:
        return np.zeros_like(img, dtype=np.float64)
    return (img - mn) / denominator

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    if old_h == 0 or old_w == 0:
        return np.zeros(new_shape)
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

def extract_bemd_features(roi):
    # Simulasi IMF1 (BEMD)
    imf1 = detrend(detrend(roi, axis=0), axis=1)
    imf1_gray = mat2gray_python(imf1)
    
    # Flatten secara Column-Major ('F') untuk padanan MATLAB
    feat = imf1_gray.flatten(order='F')
    
    # PENYELARASAN JUMLAH FEATURES (Paksa jadi 12,000)
    target_len = 12000
    if len(feat) < target_len:
        feat = np.pad(feat, (0, target_len - len(feat)), 'constant')
    else:
        feat = feat[:target_len]
    return feat.reshape(1, -1)

# --- 4. ANTARAMUKA PENGGUNA (UI) UTAMA ---
# Logo atau ikon GPR (opsional)
st.markdown("<h1>📡 GPR Intelligent Subsurface Classifier</h1>", unsafe_allow_html=True)
st.markdown("Developed for advanced GPR target identification.")
st.markdown("---")

if model is None or scaler is None:
    st.error("Ralat: Fail model `svm_model.pkl` atau `scaler.pkl` tidak dijumpai dalam repositori.")
else:
    # Sidebar
    st.sidebar.header("🕹️ Target Navigation")
    # Had dynamic berdasarkan saiz data input (dianggarkan 312x450)
    v_pos = st.sidebar.slider("Depth (Vertical)", 0, 312-105, 120)
    h_pos = st.sidebar.slider("Trace (Horizontal)", 0, 450-125, 200)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sensitivity Tuning")
    # Jika sistem masih mengesan noise sebagai objek, slider ini perlu digerakkan ke KANAN
    soil_limit = st.sidebar.slider("Background Noise Filter", 0.005, 0.030, 0.015, step=0.001)

    uploaded_files = st.file_uploader("Muat naik data GPR (.rad & .rd3)", type=["rad", "rd3"], accept_multiple_files=True)

    if len(uploaded_files) == 2:
        rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
        rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

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
            # Reshape mengikut Column-Major ('F')
            matrix = raw_data[:samples*num_traces].reshape((samples, num_traces), order='F')
            
            # Pra-pemprosesan (Background Removal & Normalization)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)
            
            # Slicing ROI (100x120)
            r_v, r_h = min(v_pos + 100, samples), min(h_pos + 120, num_traces)
            roi_crop = full_img[v_pos:r_v, h_pos:r_h]
            roi_ready = matlab_resize_manual(roi_crop, (100, 120))
            
            # Analisis Statistik Energy (Std Deviation)
            energy = np.std(roi_ready)

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Radargram View")
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0e1117') # Padankan dengan mod gelap
                ax.set_facecolor('#0e1117')
                ax.imshow(full_img, cmap='gray', aspect='auto')
                # ROI Box Merah yang jelas
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=3, edgecolor='#e74c3c', facecolor='none')
                ax.add_patch(rect)
                plt.axis('off') # Sembunyikan paksi untuk paparan lebih bersih
                st.pyplot(fig)

            with col2:
                st.subheader("Target Analysis")
                
                # --- [FIX CRITICAL] DUAL-CHECK LOGIC UNTUK PENGECAMAN HIPERBOLA ---
                
                # Pertama, semak sama ada ROI itu hanyalah noise lapisan tanah yang mendatar
                # (Sama seperti dalam `image_2.png` yang anda paparkan)
                
                # Jika energy terlalu rendah, ia hanyalah latar belakang
                if energy < soil_limit:
                    st.markdown('<div class="result-card" style="background-color: #7f8c8d;">NO TARGET ⚪</div>', unsafe_allow_html=True)
                    st.write("Status: Scanning Background Soil (Low Noise)")
                
                else:
                    # Kedua, jika energy kuat, kita semak susunan piksel (Perimeter Check)
                    # Ini digunakan untuk membezakan hiperbola daripada lapisan noise atas yang kuat
                    
                    roi_upper_middle = np.mean(roi_ready[:30, 40:80]) # Bahagian atas tengah (Apex)
                    roi_top_strip = np.mean(roi_ready[:10, :])       # Seluruh jalur paling atas
                    
                    # Logik: Pada hiperbola (Brick), isyarat atas-tengah paling kuat.
                    # Pada 'noise layer' (seperti dalam gambar anda), seluruh jalur atas adalah kuat.
                    if roi_top_strip > (roi_upper_middle * 1.5):
                        # Isyarat paling kuat berada di sepanjang jalur atas (bukan Apex tunggal)
                        st.markdown('<div class="noise-card">Scanning Noise Layer / Flat Layer</div>', unsafe_allow_html=True)
                        st.write("Detection: Texture pattern suggests a flat ground layer, not a hyperbolic target.")
                    else:
                        # Ini adalah hiperbola (kerana isyarat atas-tengah lebih kuat daripada jalur penuh)
                        
                        # Jalankan Klasifikasi SVM untuk material
                        features = extract_bemd_features(roi_ready)
                        scaled_features = scaler.transform(features)
                        prediction = model.predict(scaled_features)[0]
                        
                        # --- [FIX UI] PAPARAN KEPUTUSAN YANG KEMAS ---
                        if prediction == 1:
                            st.markdown('<div class="result-card" style="background-color: #2ecc71;">CAVITY<br>(VOID) ✅</div>', unsafe_allow_html=True)
                        elif prediction == 2:
                            st.markdown('<div class="result-card" style="background-color: #f1c40f; color: #1a1c24;">BRICK /<br>CONCRETE 🧱</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-card" style="background-color: #e74c3c;">METAL<br>PIPE ⚙️</div>', unsafe_allow_html=True)

                st.markdown("---")
                # Semak metrik ini dalam mod gelap, ia sepatutnya gelap sekarang
                st.metric(label="Reflection Energy Intensity", value=f"{energy:.4f}")
                
                st.write("BEMD ROI Input:")
                # Paparkan ROI input untuk semakan manual oleh user (mat2gray digunakan semula)
                # Tetapi jika energy terlalu rendah, paparkan skrin hitam sahaja (untuk UI yang bersih)
                display_roi = mat2gray_python(roi_ready) if energy >= 0.001 else np.zeros((100,120))
                st.image(display_roi, caption="Source ROI for AI Model", use_container_width=True)

    else:
        st.info("Sila muat naik kedua-dua fail .rad dan .rd3 untuk memulakan pengesanan.")
