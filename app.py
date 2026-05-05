import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- 1. ROBUST ASSET LOADING ---
@st.cache_resource
def load_assets():
    # This finds the exact folder where app.py is located
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR: Could not find model files in GitHub.")
        st.info("Ensure 'svm_model.pkl' and 'scaler.pkl' are uploaded to the same folder as app.py.")
        st.write(f"Debug Info: Looking for {model_path}")
        return None, None

model, scaler = load_assets()

st.set_page_config(layout="wide", page_title="GPR High-Precision Classifier")
st.title("📡 GPR Target Classifier")

# --- 2. USER INTERFACE ---
st.sidebar.header("ROI Alignment")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)

# ROI fixed to 100x120 as per gpr_bemd.xlsx requirements
BOX_H, BOX_W = 100, 120

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Parse RAD
        rad_text = rad_file.getvalue().decode("utf-8")
        samples_val = 312
        for line in rad_text.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # B. Read Binary (High Precision)
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Reshape (F-order for MALA/MATLAB)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # C. Extract ROI
            y1, x1 = min(v_start, samples_val - BOX_H), min(h_start, num_traces - BOX_W)
            roi_raw = matrix_clean[y1:y1+BOX_H, x1:x1+BOX_W]
            
            if roi_raw.size > 0:
                # Resize
                img = Image.fromarray(roi_raw)
                img_res = img.resize((BOX_W, BOX_H), resample=Image.BICUBIC)
                roi_resized = np.array(img_res, dtype=np.float64)
                
                # D. PRECISION NORMALIZATION (6+ Decimals)
                # Matches the exact statistical fingerprint of your Excel data
                # We normalize to the mean and map to the specific 0.08 amplitude limit
                roi_min, roi_max = np.min(roi_resized), np.max(roi_resized)
                if roi_max - roi_min > 0:
                    roi_scaled = ((roi_resized - roi_min) / (roi_max - roi_min) * 0.16) - 0.08
                else:
                    roi_scaled = roi_resized

                # E. PREDICT (Using Column-major flattening to match MATLAB)
                features = roi_scaled.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")

                # F. DISPLAY
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    limit = np.percentile(np.abs(matrix_clean), 98)
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    ax.add_patch(patches.Rectangle((x1, y1), BOX_W, BOX_H, color='red', fill=False, lw=2))
                    st.pyplot(fig)

                with col2:
                    st.header("Result")
                    if result == "Cavity":
                        st.success(f"### {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### {result} 🧱")
                    else:
                        st.error(f"### {result} ⚙️")
                    
                    st.write("**Data Precision Check:**")
                    st.code(f"Scale Max: {np.max(roi_scaled):.8f}")
                    
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_scaled, cmap='gray')
                    ax2.set_title("Resized Input to SVM")
                    st.pyplot(fig2)
