import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Analyzer (Fixed Hyperbola)")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary trace-by-trace
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # SUCCESS: This number 312 matches your .rad header!
    SAMPLES = 312 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape using Fortran 'F' order to stack traces vertically
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # Match MATLAB: Flip vertically and subtract mean (Background Removal)
        matrix = np.flipud(matrix)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # Classification (Resize to 100x120 to match your BEMD training)
        if TRACES >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Detection: {labels.get(pred, 'Unrecognizable')}")

        # Visualization: This will now show perfect hyperbolas
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        ax.set_title("Corrected Hyperbolic Radargram")
        st.pyplot(fig)
