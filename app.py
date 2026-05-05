import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 Mala GPR Hyperbolic Reconstruction")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary trace-by-trace
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # CRITICAL: We must match the samples per trace from your training
    # Looking at your 'Cav001.png', it uses a much higher sample count.
    # If your training used 100 samples, we must crop or resize.
    SAMPLES = 100 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape using Fortran order 'F' to keep traces vertical
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # apply vertical flip to put surface at top
        matrix = np.flipud(matrix) 
        
        # Background Removal: Subtract average of all traces to reveal hyperbolas
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 2. Classification
        # We only take the first 120 traces if that's what your SVM was trained on
        if TRACES >= 120:
            sub_matrix = matrix[:, :120]
            features = sub_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            prediction = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
            st.success(f"Classification: {labels[prediction]}")

        # 3. Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use seismic (red-white-blue) to highlight the hyperbolic phase
        v_limit = np.percentile(np.abs(matrix), 98)
        ax.imshow(matrix, cmap='seismic', aspect='auto', 
                  vmin=-v_limit, vmax=v_limit, interpolation='bilinear')
        
        ax.set_title("Reconstructed Hyperbolic Radargram")
        ax.set_ylabel("Time Samples (Depth)")
        ax.set_xlabel("Trace Number")
        st.pyplot(fig)
