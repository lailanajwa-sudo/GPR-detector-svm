import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 Mala RD3 Hyperbolic Reconstruction")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary trace-by-trace (int16 per Mala standard)
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # --- CRITICAL FIX ---
    # The 'slanted' stripes happen because this number is wrong.
    # Check your .rad file for 'samples:'. Common values: 512, 400, 1024.
    # We will try 512 first, as it is a common Mala default.
    SAMPLES = 512 
    TRACES = len(raw_data) // SAMPLES
    
    if TRACES > 0:
        # Reshape exactly like MATLAB: data = fread(fid, [samples, inf], 'int16')
        # Using 'F' (Fortran) order is mandatory to keep traces vertical
        matrix = raw_data[:SAMPLES*TRACES].reshape((SAMPLES, TRACES), order='F')
        
        # Match MATLAB logic: Flip data vertically
        matrix = np.flipud(matrix) 
        
        # Remove Background Noise (Removes horizontal air-wave stripes)
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 2. Classification Logic
        # Ensure the features extracted match your 12,000-feature training set
        if TRACES >= 120:
            # Sub-sample or Resize to match your SVM's expected input shape
            # (Assuming training was 100 samples x 120 traces)
            sub_matrix = matrix[:100, :120] 
            features = sub_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            prediction = model.predict(scaled_feat)[0]
            labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
            st.success(f"### Classification: {labels[prediction]}")

        # 3. Professional Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use seismic colormap to show hyperbolic phase changes
        v_limit = np.percentile(np.abs(matrix), 98)
        ax.imshow(matrix, cmap='seismic', aspect='auto', 
                  vmin=-v_limit, vmax=v_limit, interpolation='bilinear')
        
        ax.set_title("Reconstructed B-Scan (Hyperbolic View)")
        ax.set_ylabel("Depth (Samples)")
        ax.set_xlabel("Trace Number")
        st.pyplot(fig)
