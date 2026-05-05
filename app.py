import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA RD3 Hyperbolic Analysis")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary like MATLAB's fread(fid, ..., 'int16')
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # CRITICAL: This MUST match the 'samples' line in your .rad file
    # If the image is slanted, this number (SAMPLES) is wrong.
    # Check your .rad file! It is likely 512, 400, or 1024.
    SAMPLES = 512 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape using 'F' order to stack traces vertically (MATLAB style)
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # Match MATLAB logic: Flip data vertically (flipud)
        matrix = np.flipud(matrix) 
        
        # Background Removal: Reveals the hyperbola by removing horizontal noise
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 2. Classification
        if TRACES >= 120:
            # Sub-sample to 100x120 to match the BEMD training features
            sub_matrix = matrix_clean[:100, :120]
            features = sub_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Detection: {labels.get(pred, 'Unrecognizable')}")

        # 3. Visualization matching Cav001.png
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use seismic (Red-White-Blue) or gray to see the hyperbolic phase
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='seismic', aspect='auto', 
                  vmin=-limit, vmax=limit, interpolation='bilinear')
        
        ax.set_title("Reconstructed Radargram")
        ax.set_ylabel("Depth (Samples)")
        ax.set_xlabel("Trace Number")
        st.pyplot(fig)
