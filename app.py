import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 MALA GPR Final Classifier")

# ROI Positioners
st.sidebar.header("Target Selection")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 125) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 350, 215)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 1. Extract ROI
            roi = matrix_clean[v_start : v_start+100, h_start : h_start+120]
            
            # 2. THE CRITICAL FIX: RE-SCALE TO MATCH EXCEL DECIMAL RANGE
            # Your Excel data is in range approx -0.03 to +0.03.
            # We scale the live ROI to match that exactly.
            if np.max(np.abs(roi)) > 0:
                roi_scaled = (roi / np.max(np.abs(roi))) * 0.03
            else:
                roi_scaled = roi

            # 3. Visualization
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 98)

            with col1:
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                rect = patches.Rectangle((h_start, v_start), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                # 4. Predict using 'F' order
                features = roi_scaled.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # MAPPING: 1=Cavity, 2=Brick, 3=Metal Pipe (based on your Excel order)
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")
                
                st.subheader(f"Result: {result}")
                if result == "Cavity":
                    st.success("Detected ✅")
                else:
                    st.error("Try adjusting the ROI sliders")

    else:
        st.error("Upload files first.")
