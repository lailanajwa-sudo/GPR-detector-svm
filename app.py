import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained model
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 GPR Hyperbolic Reconstruction")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary data (int16 is standard for Mala RD3)
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # CRITICAL FIX: If your image is slanted, this number is wrong.
    # Check your .rad file for 'SAMPLES'. It is likely 512, 400, or 100.
    SAMPLES_PER_TRACE = 100 
    
    num_traces = len(raw_data) // SAMPLES_PER_TRACE
    
    if num_traces > 0:
        # Reshape using 'F' order to keep traces vertical
        matrix = raw_data[:SAMPLES_PER_TRACE * num_traces].reshape((SAMPLES_PER_TRACE, num_traces), order='F')
        
        # Match MATLAB: Flip vertically and remove background
        matrix = np.flipud(matrix) 
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # Classification
        # We use the first 120 traces (12,000 features total) as trained in Colab
        if num_traces >= 120:
            features = matrix_clean[:, :120].flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
            st.success(f"### Detected: {labels[pred]}")

        # Visualization: This will now look like your Cav001.png
        fig, ax = plt.subplots(figsize=(10, 6))
        v_limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-v_limit, vmax=v_limit)
        ax.set_title("Hyperbolic Radargram")
        ax.set_ylabel("Depth (Samples)")
        ax.set_xlabel("Traces")
        st.pyplot(fig)
