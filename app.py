import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Analyzer")

# User uploads BOTH files
uploaded_files = st.file_uploader("Upload BOTH .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = None
    rd3_file = None

    # Identify which file is which
    for f in uploaded_files:
        if f.name.endswith('.rad'):
            rad_file = f
        if f.name.endswith('.rd3'):
            rd3_file = f

    if rad_file and rd3_file:
        # 1. Read SAMPLES from .rad file
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 # Default from your provided rad file
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        st.info(f"Detected Samples per Trace: {samples_val}")

        # 2. Read Binary .rd3 file
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        traces = len(raw_data) // samples_val
        
        if traces > 0:
            # Reshape using 'F' order to keep traces vertical (as per readmala3.m)
            matrix = raw_data[:samples_val*traces].reshape((samples_val, traces), order='F')
            
            # Match MATLAB: Flip vertically and subtract mean (Background Removal)
            matrix = np.flipud(matrix)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

            # 3. Classification (Resize to 100x120 for SVM)
            if traces >= 120:
                feat_matrix = matrix_clean[:100, :120] 
                features = feat_matrix.flatten().reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                st.success(f"### Detection Result: {labels.get(pred, 'Unrecognizable')}")

            # 4. Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            limit = np.percentile(np.abs(matrix_clean), 98)
            ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
            ax.set_title(f"Radargram: {rd3_file.name}")
            st.pyplot(fig)
    else:
        st.error("Please upload both the .rad and the .rd3 file together.")
