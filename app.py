import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="GPR-X Smooth Scan", layout="wide")
st.title("📡 GPR-X Real-Time Scanner")

# Logic untuk clear file bila nak scan baru
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def trigger_new_scan():
    st.session_state.reset_key += 1
    st.rerun()

if model is None:
    st.error("⚠️ Fail model/scaler tidak dijumpai!")
else:
    # Kawasan Muat Naik (Kawasan Statik - Takkan Refresh bila slide)
    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], 
                             accept_multiple_files=True, 
                             key=f"uploader_{st.session_state.reset_key}")

    if files:
        # Butang ni hanya keluar bila dah ada fail
        st.button("➕ SCAN GAMBAR BARU", type="primary", on_click=trigger_new_scan)

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
        matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
        full_img = mat2gray_python(matrix[40:, :] - np.mean(matrix[40:, :], axis=1, keepdims=True))

        # --- 3. THE MAGIC: ST.FRAGMENT ---
        # Ini akan buatkan hanya kawasan dalam fungsi ni sahaja yang refresh
        @st.fragment
        def scan_area(img_data):
            col_preview, col_analysis = st.columns([2, 1])
            
            with col_preview:
                st.subheader("Preview Radargram")
                # Placeholder supaya gambar tak melompat
                image_spot = st.empty()
                
                # Slider diletakkan side-by-side supaya nampak kecil di bawah
                st.write("🕹️ **Manual Bounding Box Control**")
                c1, c2 = st.columns(2)
                with c1:
                    h_pos = st.slider("Trace (X)", 0, img_data.shape[1]-120, int(img_data.shape[1]/2))
                with c2:
                    v_pos = st.slider("Depth (Y)", 0, img_data.shape[0]-100, 80)

                # Render plot dengan pantas
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(img_data, cmap='gray', aspect='auto')
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                ax.add_patch(rect)
                plt.axis('off')
                image_spot.pyplot(fig)

            # --- 4. CLASSIFICATION ---
            roi_raw = img_data[v_pos:v_pos+100, h_pos:h_pos+120]
            roi_ready = matlab_resize_manual(roi_raw, (100, 120))
            energy = np.std(roi_ready)
            
            apex_idx = np.argmax(np.std(roi_ready, axis=0))
            waveform = roi_ready[:, apex_idx]
            first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
            is_cavity_phase = first_peak <= 0.50 

            if energy < 0.0135: res, color = "SOIL ⚪", "#484f58"
            elif energy > 0.026: res, color = "METAL ⚙️", "#da3633"
            else: res, color = ("CAVITY ✅", "#238636") if is_cavity_phase else ("BRICK 🧱", "#d29922")

            with col_analysis:
                st.subheader("Live Analysis")
                st.markdown(f'''<div style="padding:15px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:22px; font-weight:bold;">{res}</div>''', unsafe_allow_html=True)
                st.metric("Signal Energy", f"{energy:.4f}")
                
                # Feature BEMD
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                st.image(mat2gray_python(imf1), caption="BEMD View", use_container_width=True)

        # Panggil fungsi fragment
        scan_area(full_img)
