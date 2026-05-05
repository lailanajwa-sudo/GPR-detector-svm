import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load trained assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 MALA GPR Analyzer (RAD Input)")

# User ONLY uploads the .rad file
uploaded_rad = st.file_uploader("Upload .rad header file", type=["rad"])

# Path to the binary data stored in your GitHub repo
INTERNAL_DATA = "data.bin" 

if uploaded_rad:
    # 1. Parse .rad for SAMPLES count (Fixes slanted lines)
    rad_text = uploaded_rad.getvalue().decode("utf-8")
    samples_val = 312 # Default from your rad file
    for line in rad_text.split('\n'):
        if "SAMPLES:" in line:
            samples_val = int(line.split(':')[1].strip())
    
    st.info(f"System configured to {samples_val} samples per trace.")

    if os.path.exists(INTERNAL_DATA):
        # 2. Load the binary data from the GitHub storage
        raw_data = np.frombuffer(open(INTERNAL_DATA, "rb").read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Reshape using 'F' order to keep traces vertical 
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # Replicate MATLAB processing 
            matrix = np.flipud(matrix) # flipud(data)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True) # Background Removal

            # 3. Classification (Resize to 100x120 for SVM)
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
            ax.set_title("Hyperbolic Radargram Reconstructed")
            st.pyplot(fig)
    else:
        st.error(f"Internal data file '{INTERNAL_DATA}' not found in repository.")
