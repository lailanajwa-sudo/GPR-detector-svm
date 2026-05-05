import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 GPR Classifier (RAD Header Input)")

# User ONLY uploads the .rad file
uploaded_rad = st.file_uploader("Upload .rad file", type=["rad"])

# PATH TO YOUR DATA FILE ON GITHUB
# Make sure you upload your .rd3 file to your GitHub repo!
RD3_DATA_PATH = "data_source.rd3" 

if uploaded_rad and os.path.exists(RD3_DATA_PATH):
    # 1. Parse the uploaded .rad file for the Sample Count
    rad_content = uploaded_rad.getvalue().decode("utf-8")
    samples_per_trace = 312 # Default from your 2D_2638.rad
    for line in rad_content.split('\n'):
        if "SAMPLES:" in line:
            samples_per_trace = int(line.split(':')[1].strip())
    
    st.info(f"Header Config: {samples_per_trace} samples per trace detected.")

    # 2. Automatically load the .rd3 data from your GitHub storage
    with open(RD3_DATA_PATH, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.int16).astype(float)
    
    num_traces = len(raw_data) // samples_per_trace
    
    if num_traces > 0:
        # Reshape using 'F' order to fix slanted lines and show hyperbola
        matrix = raw_data[:samples_per_trace*num_traces].reshape((samples_per_trace, num_traces), order='F')
        
        # Match MATLAB: Flip and Background Removal
        matrix = np.flipud(matrix)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 3. Prediction (Using top 100 samples and 120 traces to match SVM features)
        if num_traces >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Classification Result: {labels.get(pred, 'Unrecognizable')}")

        # 4. Display the Hyperbolic Pattern
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        ax.set_title("Reconstructed Hyperbolic Radargram")
        st.pyplot(fig)
else:
    if not os.path.exists(RD3_DATA_PATH):
        st.warning(f"System Error: {RD3_DATA_PATH} not found in GitHub folder.")
