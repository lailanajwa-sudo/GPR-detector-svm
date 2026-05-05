import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained assets
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📡 MALA GPR Classifier")

# The user must upload both because rd3 contains the actual data 
uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next(f for f in uploaded_files if f.name.endswith('.rad'))
    rd3_file = next(f for f in uploaded_files if f.name.endswith('.rd3'))

    # 1. Parse .rad file for Sample Count 
    rad_content = rad_file.getvalue().decode("utf-8")
    samples_per_trace = 312 # Default
    for line in rad_content.split('\n'):
        if "SAMPLES:" in line:
            samples_per_trace = int(line.split(':')[1].strip())
    
    st.info(f"Header Detected: {samples_per_trace} samples per trace.")

    # 2. Process .rd3 Binary Data 
    raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
    num_traces = len(raw_data) // samples_per_trace
    
    if num_traces > 0:
        # Reshape using 'F' order to keep traces vertical 
        matrix = raw_data[:samples_per_trace*num_traces].reshape((samples_per_trace, num_traces), order='F')
        
        # Flip and Remove Background to show Hyperbolas 
        matrix = np.flipud(matrix)
        matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)

        # 3. Predict (Match training features 100x120)
        if num_traces >= 120:
            feat_matrix = matrix_clean[:100, :120] 
            features = feat_matrix.flatten().reshape(1, -1)
            scaled_feat = scaler.transform(features)
            pred = model.predict(scaled_feat)[0]
            
            labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
            st.success(f"### Result: {labels.get(pred, 'Unrecognizable')}")

        # 4. Display Radargram
        fig, ax = plt.subplots()
        limit = np.percentile(np.abs(matrix_clean), 98)
        ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
        st.pyplot(fig)
