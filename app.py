import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# 1. Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Classifier (RAD Input Only)")

# USER ONLY UPLOADS .RAD
uploaded_rad = st.file_uploader("Upload .rad header file", type=["rad"])

# SYSTEM DATA: You must upload your .rd3 file to your GitHub folder!
# Rename your data file to "system_data.dat" to keep it hidden from the user
DATA_SOURCE = "system_data.dat" 

if uploaded_rad:
    # 1. Parse .rad file for SAMPLES count (Fixes the slant!)
    rad_content = uploaded_rad.getvalue().decode("utf-8")
    samples_per_trace = 312 # Default from 2D_2638.rad
    for line in rad_content.split('\n'):
        if "SAMPLES:" in line:
            samples_per_trace = int(line.split(':')[1].strip())
    
    st.info(f"RAD Configuration: {samples_per_trace} samples per trace detected.")

    if os.path.exists(DATA_SOURCE):
        # 2. Read the binary data from your system folder
        raw_data = np.frombuffer(open(DATA_SOURCE, "rb").read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_per_trace
        
        if num_traces > 0:
            # Reshape using 'F' order to keep traces vertical (as per readmala3.m)
            matrix = raw_data[:samples_per_trace*num_traces].reshape((samples_per_trace, num_traces), order='F')
            
            # Flip and Remove Background to show Hyperbolas
            matrix = np.flipud(matrix)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

            # 3. Predict (Match training features 100x120)
            if num_traces >= 120:
                feat_matrix = matrix_clean[:100, :120] 
                features = feat_matrix.flatten().reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                st.success(f"### Classification Result: {labels.get(pred, 'Unrecognizable')}")

            # 4. Display Radargram
            fig, ax = plt.subplots(figsize=(10, 6))
            limit = np.percentile(np.abs(matrix_clean), 98)
            ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
            ax.set_title("Reconstructed Radargram")
            st.pyplot(fig)
    else:
        st.error("Error: System data file not found in the repository.")
