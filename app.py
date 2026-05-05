import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Analyzer")

# User uploads both files
uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
    rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

    # 1. Parse .rad for SAMPLES count
    rad_content = rad_file.getvalue().decode("utf-8")
    samples_val = 312 # Default from your rad file
    for line in rad_content.split('\n'):
        if "SAMPLES:" in line:
            samples_val = int(line.split(':')[1].strip())

    # 2. Read .rd3 Binary Data
    raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
    num_traces = len(raw_data) // samples_val
    
    if num_traces > 0:
        # Reshape using 'F' order to fix slanted lines
        matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
        
        # --- CRITICAL FIXES FOR VISUALS ---
        
        # A. Flip vertically to match GPR depth convention
        matrix = np.flipud(matrix) 
        
        # B. BACKGROUND REMOVAL: Subtract mean of every row
        # This removes the horizontal lines seen in your 1st image
        matrix_bg_removed = matrix - np.mean(matrix, axis=1, keepdims=True)
        
        # C. CONTRAST SCALING: Replicating mat2gray
        # We use a 98th percentile clip to make the hyperbola pop against the black background
        limit = np.percentile(np.abs(matrix_bg_removed), 98)

        # 3. Visualization (Matches Cav001.png style)
        st.subheader("Corrected Radargram")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use 'gray' colormap and set vmin/vmax to force high contrast
        img = ax.imshow(matrix_bg_removed, 
                        cmap='gray', 
                        aspect='auto', 
                        vmin=-limit, 
                        vmax=limit)
        
        ax.set_ylabel("Time/Depth (Samples)")
        ax.set_xlabel("Trace Number")
        plt.colorbar(img, ax=ax, label="Signal Intensity")
        st.pyplot(fig)

        # 4. Classification
        if num_traces >= 120:
            feat_matrix = matrix_bg_removed[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            st.success(f"### Classification Result: {labels.get(pred, 'Unrecognizable')}")
