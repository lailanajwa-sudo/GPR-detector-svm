import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 MALA RD3 Hyperbolic Analyzer")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary trace-by-trace (int16 per Mala standard)
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # FIXED: This matches the 'SAMPLES:312' in your .rad file
    SAMPLES = 312 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape using 'F' (Fortran) order to keep traces vertical (MATLAB style)
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # Match MATLAB: Flip vertically (flipud)
        matrix = np.flipud(matrix) 
        
        # Background Removal: Essential to reveal the hyperbolic shape
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 2. Classification
        # Resize/Crop to match your 12,000-feature SVM training (100 samples x 120 traces)
        if TRACES >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Classification: {labels.get(pred, 'Unrecognizable')}")

        # 3. Visualization (Matches Cav001.png target)
        st.subheader("Radargram (B-Scan)")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Contrast adjustment
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        
        ax.set_ylabel("Depth (Samples)")
        ax.set_xlabel("Trace Number")
        st.pyplot(fig)
    else:
        st.error("Incomplete data file.")
