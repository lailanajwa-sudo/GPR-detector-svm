import streamlit as st
import numpy as np
import joblib
import os
from scipy.signal import detrend
import matplotlib.pyplot as plt
from PIL import Image
import io

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

# --- 2. FAST PROCESSING ---
def get_clean_img(file_content):
    raw = np.frombuffer(file_content, dtype=np.int16).astype(np.float64)
    matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
    matrix_clean = matrix[40:, :] - np.mean(matrix[40:, :], axis=1, keepdims=True)
    img = (matrix_clean - np.min(matrix_clean)) / (np.max(matrix_clean) - np.min(matrix_clean))
    return img

# --- 3. UI SETUP ---
st.set_page_config(page_title="GPR-X Smooth", layout="wide")

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

if model is None:
    st.error("Assets missing!")
else:
    # Kawasan Muat Naik
    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], 
                             accept_multiple_files=True, 
                             key=f"u_{st.session_state.reset_key}")

    # Tombol Reset (Hanya muncul bila ada fail)
    if files:
        if st.button("🗑️ CLEAR & NEW SCAN", type="primary"):
            st.session_state.reset_key += 1
            st.rerun()

    if len(files) == 2:
        rd3_f = next(f for f in files if f.name.endswith('.rd3'))
        full_img = get_clean_img(rd3_f.getvalue())

        # --- 4. THE SMOOTH UI ---
        col_view, col_side = st.columns([2, 1])

        with col_view:
            st.subheader("Live Scanner")
            
            # Kita guna CSS untuk letak kotak hijau atas gambar
            # Ini takkan refresh gambar, cuma gerakkan 'overlay' sahaja
            h_pos = st.slider("Horizontal (Trace)", 0, full_img.shape[1]-120, int(full_img.shape[1]/2))
            v_pos = st.slider("Vertical (Depth)", 0, full_img.shape[0]-100, 80)

            # Convert numpy ke Image untuk display laju
            img_pil = Image.fromarray((full_img * 255).astype(np.uint8))
            
            # Guna HTML/CSS untuk buat box
            # Cara ni jauh lebih smooth daripada Matplotlib
            st.markdown(
                f"""
                <div style="position: relative; width: 100%; border: 1px solid #444;">
                    <img src="data:image/png;base64,{st.image(img_pil, use_container_width=True)}" style="width: 100%; display: block;">
                    <div style="
                        position: absolute;
                        border: 3px solid #00ff00;
                        left: {h_pos / full_img.shape[1] * 100}%;
                        top: {v_pos / full_img.shape[0] * 100}%;
                        width: {(120 / full_img.shape[1]) * 100}%;
                        height: {(100 / full_img.shape[0]) * 100}%;
                        pointer-events: none;
                        box-shadow: 0 0 10px #00ff00;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # --- 5. RESULT ---
        with col_side:
            st.subheader("Analysis")
            roi = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
            energy = np.std(roi)
            
            # Result box
            if energy < 0.0135: res, color = "SOIL ⚪", "#484f58"
            elif energy > 0.026: res, color = "METAL ⚙️", "#da3633"
            else: res, color = "TARGET ✅", "#238636"

            st.markdown(f'<div style="padding:20px; background:{color}; color:white; border-radius:10px; text-align:center; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
            st.metric("Energy", f"{energy:.4f}")
            
            # Small BEMD Preview
            roi_imf = detrend(detrend(roi, axis=0), axis=1)
            st.image((roi_imf - np.min(roi_imf))/(np.max(roi_imf)-np.min(roi_imf)), caption="Feature View", use_container_width=True)
