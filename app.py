import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="MALA GPR Analyzer")
st.title("📡 GPR Radargram & SVM Classifier")

uploaded_file = st.file_uploader("Upload Mala .rd3 file", type=["rd3"])

if uploaded_file:
    # 1. Read Binary like MATLAB fread(fid, [header.samples, inf], 'int16')
    # Based on your training, we assume 100 samples per trace
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # IMPORTANT: Adjust SAMPLES to match your .rad header (e.g., 512, 100, etc.)
    SAMPLES = 512 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape using 'F' (Fortran) order to keep traces vertical
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # Match MATLAB: Flip data vertically
        matrix = np.flipud(matrix) 
        
        # Background Removal (Horizontal mean subtraction)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 2. Classification
        # We need to extract the same 12,000 features your SVM expects.
        # This usually involves resizing the image to 100x120.
        if TRACES >= 120:
            # Simple slice for classification to match input shape
            # In a full version, you would apply BEMD here as in bemd_gpr.m
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            result = labels.get(pred, "Unrecognizable")
            
            st.success(f"### Detection Result: {result}")

        # 3. Visualization matching Cav001.png
        st.subheader("Radargram (B-Scan)")
        fig, ax = plt.subplots(figsize=(10, 5))
        limit = np.percentile(np.abs(matrix_clean), 98)
        im = ax.imshow(matrix_clean, cmap='gray', aspect='auto', 
                       vmin=-limit, vmax=limit)
        ax.set_ylabel("Time (samples)")
        ax.set_xlabel("Trace")
        plt.colorbar(im)
        st.pyplot(fig)
    else:
        st.error("File data too small for analysis.")
