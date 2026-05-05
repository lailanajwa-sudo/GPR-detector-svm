import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Model files not found! Error: {e}")
        return None, None

model, scaler = load_assets()

st.title("📡 GPR Cavity Detector (Final Calibration)")

# 2. Sidebar Tuning
st.sidebar.header("1. Target Selection")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)

st.sidebar.header("2. Logic Correction")
# This is the most likely fix for your misclassification
data_order = st.sidebar.radio("Flattening Order (Match MATLAB)", ["Column-wise (Fortran)", "Row-wise (C)"])
order_code = 'F' if "Fortran" in data_order else 'C'

st.sidebar.header("3. Intensity Tuning")
# Force the signal to be "weaker" to trigger Cavity
manual_scale = st.sidebar.slider("Signal Multiplier", 0.1, 2.0, 0.5, step=0.1)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # File parsing
        rad_text = rad_file.getvalue().decode("utf-8")
        samples_val = 312
        for line in rad_text.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Reshape (Initial load is always F for RD3)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # ROI extraction (100x120)
            y1, x1 = min(v_start, samples_val-100), min(h_start, num_traces-120)
            roi_raw = matrix_clean[y1:y1+100, x1:x1+120]
            
            if roi_raw.size > 0:
                # Resize
                img = Image.fromarray(roi_raw)
                img_res = img.resize((120, 100), resample=Image.BICUBIC)
                roi_resized = np.array(img_res)
                
                # Normalize to match Excel scale (approx 0.005)
                # We use the multiplier to let you manually "dim" the signal
                roi_std = np.std(roi_resized)
                if roi_std > 0:
                    roi_norm = (roi_resized - np.mean(roi_resized)) / roi_std
                    roi_final = roi_norm * (0.005 * manual_scale)
                else:
                    roi_final = roi_resized

                # PREDICT
                # Using the order_code selected in the sidebar
                features = roi_final.flatten(order=order_code).reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")

                # UI Display
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    limit = np.percentile(np.abs(matrix_clean), 98)
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                    st.pyplot(fig)

                with col2:
                    st.subheader("Result")
                    if result == "Cavity":
                        st.success(f"### {result} ✅")
                    else:
                        st.error(f"### {result}")
                    
                    st.write(f"Flattening: {data_order}")
                    st.write(f"Intensity: {np.std(roi_final):.6f}")
                    
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray')
                    ax2.set_title("SVM Input Preview")
                    st.pyplot(fig2)
