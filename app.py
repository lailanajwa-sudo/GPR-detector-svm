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

st.set_page_config(page_title="GPR Classifier", layout="wide")
st.title("📡 MALA GPR Analysis System")

# Manual Override Sliders in Sidebar
st.sidebar.header("Manual ROI Adjustment")
manual_mode = st.sidebar.checkbox("Enable Manual ROI", value=False)
v_start = st.sidebar.slider("Vertical Start", 0, 212, 50)
h_start = st.sidebar.slider("Horizontal Start", 0, 300, 150)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # 1. Parse RAD for samples
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # 2. Read Binary
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. Reshape (F-order is vital for MATLAB compatibility)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # Background Removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. Determine ROI Position
            if not manual_mode:
                # AUTO-DETECTION: Find the strongest signal area (excluding top 40 surface samples)
                search_area = np.abs(matrix_clean[40:, :])
                max_idx = np.unravel_index(np.argmax(search_area), search_area.shape)
                # Center the 100x120 window on the peak
                y_pos = max(0, max_idx[0] + 40 - 50) 
                x_pos = max(0, max_idx[1] - 60)
            else:
                y_pos, x_pos = v_start, h_start

            # Ensure ROI stays within bounds
            y_end = min(y_pos + 100, samples_val)
            x_end = min(x_pos + 120, num_traces)
            y_start = y_end - 100
            x_start = x_end - 120

            roi = matrix_clean[y_start:y_end, x_start:x_end]

            # 5. Display and Predict
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 99)

            with col1:
                st.subheader("Radargram (Red Box = ROI)")
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                rect = patches.Rectangle((x_start, y_start), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                st.subheader("Classification")
                # Show ROI
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                st.pyplot(fig2)

                # Prediction Logic
                # Use 'F' order to flatten exactly like MATLAB xlsread/xlswrite
                features = roi.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                result = labels.get(pred, "Unknown")
                
                if result == "Cavity":
                    st.success(f"### Result: {result} ✅")
                elif result == "Metal Pipe":
                    st.info(f"### Result: {result} 🛠️")
                else:
                    st.warning(f"### Result: {result} 🧱")
    else:
        st.error("Missing files.")
