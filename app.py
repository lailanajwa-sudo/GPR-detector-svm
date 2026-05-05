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

st.set_page_config(page_title="GPR Classifier", layout="wide")
st.title("📡 MALA GPR Analysis System")

# ROI Positioners in Sidebar
st.sidebar.header("ROI Alignment")
v_start = st.sidebar.slider("Vertical Pos", 0, 212, 120)
h_start = st.sidebar.slider("Horizontal Pos", 0, 350, 200)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # 1. Parse RAD
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # 2. Read Binary
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. Process Matrix
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. ROI Extraction (100x120)
            roi = matrix_clean[v_start : v_start+100, h_start : h_start+120]
            
            # --- STANDARDIZATION FIX ---
            # This makes the ROI "look" like the trained data statistically
            if np.std(roi) > 0:
                roi_std = (roi - np.mean(roi)) / np.std(roi)
            else:
                roi_std = roi

            # 5. UI Visualization
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 98)

            with col1:
                st.subheader("Radargram Selection")
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                rect = patches.Rectangle((h_start, v_start), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                st.subheader("Detection Result")
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_std, cmap='gray', aspect='auto')
                ax2.set_title("Input to Model")
                st.pyplot(fig2)

                # Predict using Fortran flattening
                features = roi_std.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # UPDATED LABELS: 1=Cavity, 2=Brick, 3=Metal Pipe
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")
                
                if result == "Cavity":
                    st.success(f"### Detected: {result} ✅")
                elif result == "Brick":
                    st.warning(f"### Detected: {result} 🧱")
                else:
                    st.info(f"### Detected: {result} ⚙️")

### Final Check for your Assignment:
# - Ensure the Red Box is centered on the hyperbola.
# - If the app detects "Brick" but the box is on a curve, try moving the box 
#   slightly up or down to find the 'peak' of the curve.
