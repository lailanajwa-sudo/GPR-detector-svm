import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load high-accuracy assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 GPR Target Classifier (Optimized)")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # 1. Read binary exactly like MATLAB's fread
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    if len(raw_data) >= 12000:
        # 2. Reshape [Samples, Traces] using Fortran order 'F'
        # This aligns the data vertically to form hyperbolas
        matrix = raw_data[:12000].reshape((100, 120), order='F')
        
        # 3. Flip vertically like MATLAB's flipud(data)
        matrix = np.flipud(matrix)

        # 4. Background Removal (Standard GPR processing)
        # This clarifies the hyperbolic shape for the SVM
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 5. Predict using High-Accuracy Parameters
        # Flatten to match the training input format
        feature_vector = matrix_clean.flatten().reshape(1, -1)
        scaled_feat = scaler.transform(feature_vector)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        result = labels.get(prediction, "Unknown")

        # --- Visual Result ---
        st.success(f"### Classification: {result} ({max(prob)*100:.2f}%)")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Seismic colormap helps distinguish Cavity phase from Metal phase
        limit = np.percentile(np.abs(matrix_clean), 97)
        ax.imshow(matrix_clean, cmap='seismic', aspect='auto', 
                  vmin=-limit, vmax=limit, interpolation='bilinear')
        
        ax.set_title(f"Radargram View: {result}")
        ax.set_xlabel("Trace Number")
        ax.set_ylabel("Time Sample (Depth)")
        st.pyplot(fig)
    else:
        st.error("Data size mismatch.")
