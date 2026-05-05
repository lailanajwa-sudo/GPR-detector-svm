import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load trained assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="GPR ROI Classifier", layout="wide")
st.title("📡 MALA GPR Analysis System")

# Sidebar for ROI Positioning
st.sidebar.header("ROI Controls")
st.sidebar.write("Move these sliders to box the hyperbola.")
start_sample = st.sidebar.slider("Vertical Start (Sample)", 0, 212, 50)
start_trace = st.sidebar.slider("Horizontal Start (Trace)", 0, 300, 50)

uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # 1. Parse .rad for SAMPLES
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # 2. Read Binary Data
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. Reshape (order='F' for MATLAB compatibility)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # Background Removal (Subtract Row Mean)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. Extract ROI based on Sliders
            # Window size is 100 samples x 120 traces to match training
            roi = matrix_clean[start_sample : start_sample + 100, 
                               start_trace  : start_trace  + 120]
            
            # 5. Visualization (Full Radargram with ROI Box)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Full Radargram")
                limit = np.percentile(np.abs(matrix_clean), 99)
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                
                # Draw the Red ROI Box on the main image
                rect = patches.Rectangle((start_trace, start_sample), 120, 100, 
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                st.subheader("Classification")
                # Show what the SVM is looking at
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                ax2.set_title("Selected ROI (100x120)")
                st.pyplot(fig2)

                # PREDICT
                if roi.shape == (100, 120):
                    # Use order='F' to match MATLAB flattening
                    features = roi.flatten(order='F').reshape(1, -1)
                    scaled_feat = scaler.transform(features)
                    pred = model.predict(scaled_feat)[0]
                    
                    labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                    result = labels.get(pred, "Unknown")
                    
                    if result == "Cavity":
                        st.success(f"### Detected: {result} ✅")
                    else:
                        st.warning(f"### Detected: {result}")
                else:
                    st.error("ROI window out of bounds! Adjust sliders.")
    else:
        st.error("Please upload both a .rad and a .rd3 file.")
