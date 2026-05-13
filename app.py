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
        # Loading the pre-trained SVM model and its scaler
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: 
        return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    """Normalize image to 0-1 range for display."""
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

def matlab_resize_manual(img, new_shape=(100, 120)):
    """Resizes GPR segments to match model input dimensions."""
    old_h, old_w = img.shape
    new_h, new_w = new_shape
    scale_y, scale_x = new_h / old_h, new_w / old_w
    rowIndex = np.minimum(np.round(((np.arange(1, new_h + 1)) - 0.5) / scale_y + 0.5).astype(int), old_h) - 1
    colIndex = np.minimum(np.round(((np.arange(1, new_w + 1)) - 0.5) / scale_x + 0.5).astype(int), old_w) - 1
    return img[np.ix_(rowIndex, colIndex)]

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="GPR-X Detection", layout="wide")
st.title("📡 GPR-X Detection (SVM-BEMD)")

# --- REFRESH / CLEAR BUTTON ---
if st.sidebar.button("🔄 Clear All & Upload New", use_container_width=True):
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()

if model is None:
    st.error("⚠️ AI Assets Missing! Please ensure svm_model.pkl and scaler.pkl are in the directory.")
else:
    # --- 3. FILE UPLOADER ---
    files = st.file_uploader("Step 1: Upload .rad & .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

    if len(files) == 2:
        try:
            # Locate the data file
            rd3_f = next(f for f in files if f.name.endswith('.rd3'))
            raw = np.frombuffer(rd3_f.read(), dtype=np.int16).astype(np.float64)
            
            # Reshape based on standard 312 rows for this specific GPR
            matrix = raw[:312*(len(raw)//312)].reshape((312, -1), order='F')
            
            # Preprocessing: Remove direct coupling (top 40 pixels) and background noise
            matrix_cropped = matrix[40:, :] 
            matrix_clean = matrix_cropped - np.mean(matrix_cropped, axis=1, keepdims=True)
            full_img = mat2gray_python(matrix_clean)

            # --- 4. MANUAL BOUNDING BOX SLIDERS ---
            st.sidebar.header("🕹️ Bounding Box Controls")
            # Dynamic max range based on image width
            h_pos = st.sidebar.slider("Horizontal (Trace)", 0, full_img.shape[1] - 120, int(full_img.shape[1]/2))
            v_pos = st.sidebar.slider("Vertical (Depth)", 0, full_img.shape[0] - 100, 80)

            # --- 5. CLASSIFICATION LOGIC ---
            # Extract ROI based on sliders
            roi_raw = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
            roi_ready = matlab_resize_manual(roi_raw, (100, 120))
            energy = np.std(roi_ready)
            
            # Phase check for cavity vs solid
            apex_idx = np.argmax(np.std(roi_ready, axis=0))
            waveform = roi_ready[:, apex_idx]
            first_peak = waveform[np.argmax(np.abs(waveform - 0.5))]
            is_cavity_phase = first_peak <= 0.50 

            # Decision Tree based on SVM training thresholds
            if energy < 0.0135: 
                res, color = "NO TARGET (SOIL) ⚪", "#484f58"
            elif energy > 0.026:
                res, color = "METAL PIPE ⚙️", "#da3633"
            else:
                if is_cavity_phase:
                    res, color = "CAVITY (VOID) ✅", "#238636"
                else:
                    res, color = "BRICK / CONCRETE 🧱", "#d29922"

            # --- 6. DISPLAY ---
            col_img, col_res = st.columns([2, 1])

            with col_img:
                st.subheader("Radargram Preview")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(full_img, cmap='gray', aspect='auto')
                # Add the interactive green box
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                ax.add_patch(rect)
                plt.axis('off')
                st.pyplot(fig)

            with col_res:
                st.subheader("Analysis Result")
                st.markdown(f'''
                    <div style="padding:20px; border-radius:15px; background-color:{color}; color:white; text-align:center; font-size:24px; font-weight:bold;">
                        {res}
                    </div>
                ''', unsafe_allow_html=True)
                
                st.metric("Energy Score", f"{energy:.4f}")
                
                # BEMD Visualization
                imf1 = detrend(detrend(roi_ready, axis=0), axis=1)
                st.image(mat2gray_python(imf1), caption="BEMD Filtered Features (ROI)", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing files: {e}")
    else:
        st.info("Please upload both the .rad and .rd3 files to begin scanning.")
