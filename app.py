import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load trained assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Model files (.pkl) not found. Please upload them to GitHub.")
        return None, None

model, scaler = load_assets()

st.title("📡 MALA GPR Classifier (Stable Version)")

# 1. Sidebar ROI Controls with safe limits
st.sidebar.header("Target Selection")
# Default values set to likely hyperbola locations
v_start = st.sidebar.slider("Vertical (Depth)", 0, 200, 100) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 300, 150)

uploaded_files = st.file_uploader("Upload .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # Read RAD
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # Read Binary
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 120:
            # Reshape (F-order)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 2. SAFE ROI EXTRACTION (Fixes the ValueError)
            # Ensure we don't go out of bounds
            y1 = min(v_start, samples_val - 100)
            x1 = min(h_start, num_traces - 120)
            roi = matrix_clean[y1 : y1+100, x1 : x1+120]

            # 3. PRE-PROCESSING & NORMALIZATION
            # Check if ROI is valid/not empty
            if roi.size > 0 and np.max(np.abs(roi)) > 0:
                # Scale to match the tiny decimal range of your Excel file (-0.03 to 0.03)
                roi_scaled = (roi / np.max(np.abs(roi))) * 0.03
                
                # 4. Predict
                features = roi_scaled.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # Mapping: 1=Cavity, 2=Brick, 3=Metal Pipe
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")
            else:
                result = "Invalid ROI"
                roi_scaled = roi

            # 5. Visualization
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 98)

            with col1:
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                rect = patches.Rectangle((x1, y1), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                st.subheader("Result")
                if result == "Cavity":
                    st.success(f"Detected: {result} ✅")
                elif result == "Brick":
                    st.warning(f"Detected: {result} 🧱")
                else:
                    st.info(f"Detected: {result} ⚙️")
                
                # Show normalized preview
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_scaled, cmap='gray', aspect='auto')
                ax2.set_title("Input Sample")
                st.pyplot(fig2)
        else:
            st.error("Data too small. Need at least 120 traces.")
    else:
        st.error("Missing .rad or .rd3 file.")
