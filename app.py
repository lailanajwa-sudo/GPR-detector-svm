import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Analyzer")

# 1. Improved File Uploader Logic
uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    # Initialize variables to avoid NameError
    rad_file = None
    rd3_file = None

    # Correctly assign files from the list
    for f in uploaded_files:
        if f.name.endswith('.rad'):
            rad_file = f
        elif f.name.endswith('.rd3'):
            rd3_file = f

    if rad_file and rd3_file:
        # 2. Parse .rad for SAMPLES count (312 in your file)
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        st.info(f"Header Detected: {samples_val} samples per trace.")

        # 3. Read .rd3 Binary Data
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 4. Reshape with 'F' order 
            # IMPORTANT: We REMOVE np.flipud() to fix the "terbalik" issue
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # 5. Background Removal (Subtract Mean) to match Cav001.png
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 6. Set Contrast (99th percentile for clean black/white look)
            limit = np.percentile(np.abs(matrix_clean), 99)

            # 7. Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Display original orientation (Surface at the top)
            img = ax.imshow(matrix_clean, 
                            cmap='gray', 
                            aspect='auto', 
                            vmin=-limit, 
                            vmax=limit)
            
            # Match Labels from Cav001.png
            ax.set_ylabel("Time (samples)")
            ax.set_xlabel("Trace")
            ax.set_title("Radargram Reconstructed")
            st.pyplot(fig)

            # 8. Classification Logic
            if num_traces >= 120:
                feat_matrix = matrix_clean[:100, :120] 
                features = feat_matrix.flatten().reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                st.success(f"### Classification Result: {labels.get(pred, 'Unrecognizable')}")
    else:
        st.error("Please ensure you uploaded one .rad file and one .rd3 file.")
