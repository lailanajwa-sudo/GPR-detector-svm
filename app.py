import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Classifier")

# To handle the "only rad" request, we allow multiple files but process them together
uploaded_files = st.file_uploader("Drag & Drop your .rad and .rd3 files here", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
    rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

    # 1. Parse .rad file to find SAMPLES count (Fixes the "slanted" lines)
    rad_content = rad_file.getvalue().decode("utf-8")
    samples_per_trace = 312 # Default from your 2D_2638.rad
    for line in rad_content.split('\n'):
        if "SAMPLES:" in line:
            samples_per_trace = int(line.split(':')[1].strip())
    
    st.info(f"Header Config: {samples_per_trace} samples detected.")

    # 2. Process .rd3 Binary Data trace-by-trace
    raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
    num_traces = len(raw_data) // samples_per_trace
    
    if num_traces > 0:
        # Reshape using 'F' (Fortran) order to keep traces vertical (MATLAB style)
        matrix = raw_data[:samples_per_trace*num_traces].reshape((samples_per_trace, num_traces), order='F')
        
        # Match MATLAB: Flip vertically and subtract mean (Background Removal)
        matrix = np.flipud(matrix)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 3. Prediction (Match 100x120 training features)
        if num_traces >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Classification: {labels.get(pred, 'Unrecognizable')}")

        # 4. Display Hyperbolic Radargram
        fig, ax = plt.subplots(figsize=(10, 6))
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        ax.set_title(f"Reconstructed Hyperbola: {rd3_file.name}")
        st.pyplot(fig)
