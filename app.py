import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. ASSET LOADING (CACHED) ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

# Cache pemprosesan gambar supaya tak buat kerja dua kali
@st.cache_data
def process_gpr_data(file_content):
    raw = np.frombuffer(file_content, dtype=np.int16).astype(np.float64)
    matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
    matrix_cropped = matrix[40:, :] 
    matrix_clean = matrix_cropped - np.mean(matrix_cropped, axis=1, keepdims=True)
    
    mn, mx = np.min(matrix_clean), np.max(matrix_clean)
    diff = mx - mn
    return (matrix_clean - mn) / diff if diff > 1e-7 else np.zeros_like(matrix_clean)

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 2. UI SETTINGS ---
st.set_page_config(page_title="GPR-X", layout="wide")

# Reset Function
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_files():
    st.session_state.uploader_key += 1
    st.rerun()

# --- 3. MAIN APP ---
st.title("📡 GPR-X Fast Scanner")

files = st.file_uploader("Muat naik .rad & .rd3", type=["rad", "rd3"], 
                         accept_multiple_files=True, 
                         key=f"u_{st.session_state.uploader_key}")

if files:
    if st.button("🗑️ SCAN BARU", type="secondary"):
        clear_files()

if len(files) == 2:
    rd3_f = next(f for f in files if f.name.endswith('.rd3'))
    # Guna cache supaya tak refresh data mentah
    full_img = process_gpr_data(rd3_f.getvalue())

    # --- 4. LAYOUT & SLIDERS (Kecil di bawah) ---
    col_plot, col_res = st.columns([2, 1])

    with col_plot:
        # Gunakan container kosong untuk elak flickering
        main_display = st.empty()
        
        st.write("---")
        # Slider disusun rapat
        c1, c2 = st.columns(2)
        with c1:
            h_pos = st.slider("Trace", 0, full_img.shape[1]-120, int(full_img.shape[1]/2), key="h")
        with c2:
            v_pos = st.slider("Depth", 0, full_img.shape[0]-100, 80, key="v")

        # Lukis gambar secara manual (Laju)
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Buang whitespace
        ax.imshow(full_img, cmap='gray', aspect='auto')
        rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
        ax.add_patch(rect)
        ax.axis('off')
        main_display.pyplot(fig, clear_figure=True)
        plt.close(fig) # Tutup plot supaya tak makan RAM

    # --- 5. ANALISIS ---
    roi = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
    roi_ready = matlab_resize_manual(roi, (100, 120))
    energy = np.std(roi_ready)
    
    # Polarity logic
    apex_idx = np.argmax(np.std(roi_ready, axis=0))
    waveform = roi_ready[:, apex_idx]
    first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
    is_cavity = first_peak <= 0.50 

    if energy < 0.0135: res, color = "SOIL ⚪", "#484f58"
    elif energy > 0.026: res, color = "METAL ⚙️", "#da3633"
    else: res, color = ("CAVITY ✅", "#238636") if is_cavity else ("BRICK 🧱", "#d29922")

    with col_res:
        st.subheader("Scan Result")
        st.markdown(f'<div style="padding:15px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:22px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
        st.metric("Energy Score", f"{energy:.4f}")
        
        # Detrend & BEMD Preview
        imf_img = detrend(detrend(roi_ready, axis=0), axis=1)
        st.image((imf_img - np.min(imf_img))/(np.max(imf_img)-np.min(imf_img)), caption="BEMD Feature", use_container_width=True)
