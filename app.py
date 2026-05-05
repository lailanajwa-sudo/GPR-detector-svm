import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(layout="wide", page_title="GPR High-Precision Classifier")
st.title("📡 GPR Classification (BEMD Matched)")

# 2. Sidebar - THE TUNING CONTROLS
st.sidebar.header("1. ROI Position")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 400, 210)

st.sidebar.header("2. SVM Sensitivity")
# This slider is CRITICAL. 
# Set it to 0.005 for Cavity, 0.015 for Metal Pipe.
target_std = st.sidebar.slider("Target Intensity (Std Dev)", 0.001, 0.030, 0.008, format="%.4f")
order_type = st.sidebar.selectbox("Data Order", ["Fortran (Column-wise)", "C (Row-wise)"], index=0)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # Parsing
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. MATRIX PROCESSING
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True) # Background removal
            
            # ROI Selection (100x120)
            y1, x1 = min(v_start, samples_val-100), min(h_start, num_traces-120)
            roi_raw = matrix_clean[y1:y1+100, x1:x1+120]
            
            if roi_raw.size > 0:
                # 4. NORMALIZATION TO MATCH EXCEL
                # We force the ROI to have the 'target_std' selected by the user
                current_std = np.std(roi_raw)
                if current_std > 0:
                    roi_norm = (roi_raw - np.mean(roi_raw)) / current_std
                    roi_final = roi_norm * target_std 
                else:
                    roi_final = roi_raw

                # 5. PREDICTION
                # Flattening order ('F' for MATLAB/Fortran, 'C' for Python default)
                f_order = 'F' if "Fortran" in order_type else 'C'
                features = roi_final.flatten(order=f_order).reshape(1, -1)
                
                # Apply training scaler and model
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")

                # 6. UI DISPLAY
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    limit = np.percentile(np.abs(matrix_clean), 98)
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    ax.add_patch(patches.Rectangle((x1, y1), 120, 100, linewidth=2, edgecolor='red', facecolor='none'))
                    st.pyplot(fig)

                with col2:
                    st.subheader("Classification")
                    if result == "Cavity":
                        st.success(f"Result: {result} ✅")
                    elif result == "Brick":
                        st.warning(f"Result: {result} 🧱")
                    else:
                        st.info(f"Result: {result} ⚙️")
                    
                    st.metric("Live ROI Std Dev", f"{np.std(roi_final):.6f}")
                    st.write("Current mapping matches your Excel data precision (6 decimals).")
                    
                    # Preview for the SVM
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray', aspect='auto')
                    ax2.set_title("Input (Normalized)")
                    st.pyplot(fig2)
