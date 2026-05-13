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
    except: 
        return None, None

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
st.set_page_config(page_title="GPR-X Real-Time", layout="wide")
st.title("📡 GPR-X Real-Time Detector")

# File Reset Logic
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

def clear_files():
    st.session_state['uploader_key'] += 1
    st.rerun()

if model is None:
    st.error("⚠️ AI Assets Missing!")
else:
    files = st.file_uploader("Upload .rad & .rd3", type=["rad", "rd3"], 
                             accept_multiple_files=True, 
                             key=f"u_{st.session_state['uploader_key']}")

    if files:
        st.button("➕ NEW SCAN", on_click=clear_files)

    if len(files) == 2:
        try:
            rd3_f = next(f for f in files if f.name.endswith('.rd3'))
            raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
            matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
            full_img = mat2gray_python(matrix[40:, :] - np.mean(matrix[40:, :], axis=1, keepdims=True))

            # --- 3. LAYOUT ---
            col_left, col_right = st.columns([2, 1])

            with col_left:
                # 1. The Image Preview (Top)
                st.subheader("Live Preview")
                plot_placeholder = st.empty()
                
                # 2. The Small Sliders (Bottom)
                st.write("🔍 **Position Tuning**")
                # Creating 4 columns to make sliders "smaller" (using center columns)
                _, s1, s2, _ = st.columns([0.1, 1, 1, 0.1])
                with s1:
                    h_pos = st.slider("Trace (X)", 0, full_img.shape[1]-120, int(full_img.shape[1]/2), label_visibility="collapsed")
                    st.caption("Horizontal (X)")
                with s2:
                    v_pos = st.slider("Depth (Y)", 0, full_img.shape[0]-100, 80, label_visibility="collapsed")
                    st.caption("Vertical (Y)")

                # Render Plot
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(full_img, cmap='gray', aspect='auto')
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                ax.add_patch(rect)
                plt.axis('off')
                plot_placeholder.pyplot(fig)

            # --- 4. CLASSIFICATION (Auto-updates with sliders) ---
            roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
            roi_ready = matlab_resize_manual(roi_raw, (100, 120))
            energy = np.std(roi_ready)
            
            # Polarity Logic
            apex_idx = np.argmax(np.std(roi_ready, axis=0))
            waveform = roi_ready[:, apex_idx]
            first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
            is_cavity_phase = first_peak <= 0.50 

            if energy < 0.0135: res, color = "SOIL ⚪", "#484f58"
            elif energy > 0.026: res, color = "METAL ⚙️", "#da3633"
            else: res, color = ("CAVITY ✅", "#238636") if is_cavity_phase else ("BRICK 🧱", "#d29922")

            with col_right:
                st.subheader("Live Result")
                st.markdown(f'<div style="padding:15px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:22px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
                st.metric("Energy", f"{energy:.4f}")
                
                # BEMD Feature Extract
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                st.image(mat2gray_python(imf1), caption="BEMD Texture", use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
