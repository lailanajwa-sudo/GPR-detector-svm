import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 Professional GPR Radargram Analysis")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    if len(raw_data) >= 12000:
        # --- THE FIX FOR HYPERBOLIC PATTERN ---
        # GPR data is usually stored Trace 1 (100 samples), then Trace 2, etc.
        # Reshaping with order='F' ensures each trace is a COLUMN.
        matrix = raw_data[:12000].reshape((100, 120), order='F')
        
        # --- Signal Processing ---
        # 1. Subtract mean (Background removal) to sharpen hyperbolas
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
        
        # 2. Time Gain (Boost lower signal)
        gain = np.linspace(1, 3, 100).reshape(-1, 1)
        matrix = matrix * gain

        # --- Classification ---
        feat = matrix.flatten().reshape(1, -1)
        scaled_feat = scaler.transform(feat)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}

        # --- Display ---
        st.success(f"Detected: {labels[prediction]} ({max(prob)*100:.2f}%)")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Use 'gray' or 'seismic' and adjust contrast
        v_limit = np.max(np.abs(matrix)) * 0.5 
        im = ax.imshow(matrix, cmap='gray', aspect='auto', 
                       vmin=-v_limit, vmax=v_limit, interpolation='sinc')
        
        ax.set_title("B-Scan Radargram (Hyperbolic View)")
        ax.set_xlabel("Traces (Position)")
        ax.set_ylabel("Samples (Depth)")
        plt.colorbar(im)
        st.pyplot(fig)
    else:
        st.error("Data too short.")
