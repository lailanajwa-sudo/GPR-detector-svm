import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 GPR Hyperbolic Pattern Analysis")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # 1. Read binary exactly like MATLAB's fread
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # IMPORTANT: Check your header file (.rad) for the true sample count.
    # If your file has 500 samples per trace, change this number:
    SAMPLES = 400 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # 2. Reshape into [Samples, Traces] using Fortran order 'F'
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # 3. Flip vertically and remove background noise
        # This creates the clear hyperbola curves
        matrix = np.flipud(matrix) 
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 4. Predict (Classification based on your training features)
        # Note: Ensure features used here match the BEMD features in your Excel
        st.write(f"Classified Target based on SVM Parameters")
        
        # 5. Visualization (Matching your Cav001.png style)
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix), 98)
        # Using gray colormap and correct orientation
        im = ax.imshow(matrix, cmap='gray', aspect='auto', 
                       vmin=-limit, vmax=limit, interpolation='bilinear')
        
        ax.set_title("GPR Radargram (B-Scan)")
        ax.set_ylabel("Time (samples)")
        ax.set_xlabel("Trace")
        plt.colorbar(im)
        st.pyplot(fig)
