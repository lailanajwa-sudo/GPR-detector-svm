import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA RD3 Hyperbolic Analysis")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # 1. Read binary exactly like MATLAB's fread(fid, ..., 'int16')
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # CRITICAL: This must match 'header.samples' from your .rad file
    # If the image is slanted, change this number (e.g., 512, 400, or 1024)
    SAMPLES = 512 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # 2. Reshape with 'F' order to keep traces vertical (MATLAB style)
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # 3. Flip vertically like MATLAB's flipud(data)
        matrix = np.flipud(matrix)

        # 4. Background Removal (Standard GPR processing to reveal hyperbolas)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 5. Prediction
        # Ensure input matches the 12,000 features your SVM expects
        if TRACES >= 120:
            features = matrix_clean[:100, :120].flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            prediction = model.predict(scaled_feat)[0]
            labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
            st.success(f"### Classification: {labels[prediction]}")

        # 6. Visualization (Matches your Cav001.png target)
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', 
                  vmin=-limit, vmax=limit, interpolation='bilinear')
        
        ax.set_title("Reconstructed Hyperbolic Radargram")
        ax.set_ylabel("Depth (Samples)")
        ax.set_xlabel("Trace Number")
        st.pyplot(fig)
