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
st.set_page_config(page_title="GPR-X Detection SVM", layout="wide")
st.title("📡 GPR-X Detection (SVM-BEMD)")

if model is None:
    st.error("⚠️ AI Assets Missing!")
else:
    # --- LEFT SIDEBAR (SLIDERS) ---
    with st.sidebar:
        st.header("🎯 Target Selection")
        # These are always visible on the left
        h_pos = st.slider("Horizontal Trace (X)", 0, 1000, 200)
        v_pos = st.slider("Vertical Depth (Y)", 0, 312-40-100, 80)
        
        st.divider()
        if st.button("🔄 Reset App & Clear Files", use_container_width=True):
            st.rerun()

    # --- 3. FILE UPLOADER WITH AUTO-REPLACE LOGIC ---
    # We use a session state key to track the "current" scan
    uploaded_files = st.file_uploader(
        "Upload .rad & .rd3 files (Upload new ones to replace the old ones)", 
        type=["rad", "rd3"], 
        accept_multiple_files=True
    )

    # Logic to ensure only the LATEST two files are used
    if len(uploaded_files) > 2:
        st.warning("Multiple scans detected. Using only the 2 most recently uploaded files.")
        # We take the last two files uploaded
        active_files = uploaded_files[-2:]
    else:
        active_files = uploaded_files

    if len(active_files) == 2:
        try:
            # Check if we have one of each required type
            rd3_f = next((f for f in active_files if f.name.endswith('.rd3')), None)
            rad_f = next((f for f in active_files if f.name.endswith('.rad')), None)

            if not rd3_f or not rad_f:
                st.error("Please upload one .rd3 and one .rad file.")
            else:
                raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
                matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
                
                # Remove top 40 (Air-Soil interface)
                matrix_cropped = matrix[40:, :] 
                matrix_clean = matrix_cropped - np.mean(matrix_cropped, axis=1, keepdims=True)
                full_img = mat2gray_python(matrix_clean)

                # --- 4. DISPLAY & ANALYSIS ---
                col_img, col_res = st.columns([2, 1])

                with col_img:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(full_img, cmap='gray', aspect='auto')
                    # This box moves when you move the left sliders
                    rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                    ax.add_patch(rect)
                    plt.axis('off')
                    st.pyplot(fig)

                # --- 5. CLASSIFICATION ---
                roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
                
                if roi_raw.shape[0] >= 10 and roi_raw.shape[1] >= 10:
                    roi_ready = matlab_resize_manual(roi_raw, (100, 120))
                    energy = np.std(roi_ready)
                    
                    apex_idx = np.argmax(np.std(roi_ready, axis=0))
                    waveform = roi_ready[:, apex_idx]
                    first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
                    is_cavity_phase = first_peak <= 0.50 

                    if energy < 0.0135: 
                        res, color = "NO TARGET (SOIL) ⚪", "#484f58"
                    elif energy > 0.026:
                        res, color = "METAL PIPE ⚙️", "#da3633"
                    else:
                        if is_cavity_phase:
                            res, color = "CAVITY (VOID) ✅", "#238636"
                        else:
                            res, color = "BRICK / CONCRETE 🧱", "#d29922"

                    with col_res:
                        st.subheader("Analysis")
                        st.markdown(f'<div style="padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:22px; font-weight:bold;">{res}</div>', unsafe_allow_html=True)
                        st.metric("Energy Score", f"{energy:.4f}")
                        
                        imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                        st.image(mat2gray_python(imf1), caption="BEMD Filtered View", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {e}")
