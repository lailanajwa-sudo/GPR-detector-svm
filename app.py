import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Analyzer")
st.write("Upload both .rad and .rd3 files to view the radargram and classify the target.")

# Accept multiple files
uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
    rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

    # 1. Parse .rad for SAMPLES count
    rad_text = rad_file.getvalue().decode("utf-8")
    samples_val = 312 # Default
    for line in rad_text.split('\n'):
        if "SAMPLES:" in line:
            samples_val = int(line.split(':')[1].strip())
    
    st.info(f"Header Detected: {samples_val} samples per trace.")

    # 2. Read .rd3 Binary Data
    raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
    num_traces = len(raw_data) // samples_val
    
    if num_traces > 0:
        # Reshape using 'F' order to fix slanted lines
        matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
        
        # Match MATLAB: Flip and Background Removal
        matrix = np.flipud(matrix)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 3. Classification (Resize to 100x120 for SVM)
        if num_traces >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            st.success(f"### Classification Result: {labels.get(pred, 'Unrecognizable')}")

        # 4. Show Radargram
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        ax.set_title(f"Radargram: {rd3_file.name}")
        st.pyplot(fig)
