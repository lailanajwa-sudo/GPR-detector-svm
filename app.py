import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="GPR Hyperbola Classifier", layout="wide")
st.title("📡 MALA GPR Analysis System")

uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # 1. Parse .rad for SAMPLES count
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # 2. Read Binary Data
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. Reshape (order='F' keeps traces vertical)
            # We do NOT use flipud to keep surface at the top
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # Background Removal (Subtract Row Mean)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. ROI & Prediction Logic
            # Change these to match your MATLAB ROI exactly
            sample_start, sample_end = 0, 100
            trace_start, trace_end = 0, 120

            if num_traces >= trace_end:
                # CROP TO ROI
                roi = matrix_clean[sample_start:sample_end, trace_start:trace_end]
                
                # FLATTEN (order='F' matches MATLAB column-major)
                features = roi.flatten(order='F').reshape(1, -1)
                
                # PREDICT
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                result = labels.get(pred, "Unknown")
                
                # Display Results
                st.subheader("Classification Result")
                if result == "Cavity":
                    st.success(f"### Detected: {result} ✅")
                else:
                    st.warning(f"### Detected: {result}")

            # 5. Visualization
            st.subheader("Radargram Preview")
            col1, col2 = st.columns(2)
            
            with col1:
                # Main Radargram
                limit = np.percentile(np.abs(matrix_clean), 99)
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                ax.set_title("Full Radargram (Background Removed)")
                st.pyplot(fig)

            with col2:
                # ROI Preview (What the SVM sees)
                if num_traces >= trace_end:
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    ax2.set_title("ROI Window (100x120)")
                    st.pyplot(fig2)
    else:
        st.error("Please upload both a .rad and a .rd3 file.")
