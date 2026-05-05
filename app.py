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

st.set_page_config(page_title="GPR Assignment Final", layout="wide")
st.title("📡 MALA GPR Classification System")

# ROI Positioners
st.sidebar.header("Target Selection")
# Adjust these sliders until the RED BOX is exactly on the hyperbola
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 125) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 300, 215)

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
            # Background removal (MATLAB style)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. Extract and Standardize ROI
            roi = matrix_clean[v_start:v_start+100, h_start:h_start+120]
            
            # --- THE FINAL FIX: Z-SCORE STANDARDIZATION ---
            # This replicates MATLAB's zscore(data) inside the ROI
            if np.std(roi) > 0:
                roi_final = (roi - np.mean(roi)) / np.std(roi)
            else:
                roi_final = roi

            # 5. UI Layout
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 98)

            with col1:
                st.subheader("Radargram (Target Selection)")
                fig, ax = plt.subplots()
                ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                rect = patches.Rectangle((h_start, v_start), 120, 100, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                st.pyplot(fig)

            with col2:
                st.subheader("Classification")
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_final, cmap='gray', aspect='auto')
                ax2.set_title("Standardized ROI")
                st.pyplot(fig2)

                # 6. Predict with Fortran Flattening
                features = roi_final.flatten(order='F').reshape(1, -1)
                
                # Apply the scaler from training
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # --- DOUBLE CHECK THIS MAPPING ---
                # Based on your prompt: 1=Cavity, 2=Metal Pipe, 3=Brick
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                result = labels.get(pred, "Unknown")
                
                if result == "Cavity":
                    st.success(f"### Result: {result} ✅")
                elif result == "Metal Pipe":
                    st.info(f"### Result: {result} ⚙️")
                else:
                    st.warning(f"### Result: {result} 🧱")
                    st.write("Tip: If it still says Brick, check if your Excel labels match 1=Cavity.")
