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

st.set_page_config(page_title="GPR Assignment", layout="wide")
st.title("📡 MALA GPR Classification System")

# ROI Positioners
st.sidebar.header("Target Selection")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 110) # Defaulted closer to your image
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 300, 210)

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
            # 3. Process Matrix (Fortran Order)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. Extract ROI
            roi = matrix_clean[v_start:v_start+100, h_start:h_start+120]

            # --- THE FIX: ROI NORMALIZATION ---
            # This forces the hyperbola to have the same "brightness" as your training data
            if np.max(np.abs(roi)) > 0:
                roi_norm = roi / np.max(np.abs(roi))
            else:
                roi_norm = roi

            # 5. UI Layout
            col1, col2 = st.columns([2, 1])
            limit = np.percentile(np.abs(matrix_clean), 99)

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
                ax2.imshow(roi_norm, cmap='gray', aspect='auto', vmin=-1, vmax=1)
                ax2.set_title("Normalized ROI")
                st.pyplot(fig2)

                # Prediction with 'F' flattening
                features = roi_norm.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # Check labels match your Excel order!
                # 1=Cavity, 2=Metal Pipe, 3=Brick
                labels = {1: "Cavity", 2: "Metal Pipe", 3: "Brick"}
                result = labels.get(pred, "Unknown")
                
                if result == "Cavity":
                    st.success(f"### Result: {result} ✅")
                else:
                    st.error(f"### Result: {result}")
                    st.write("Tip: Adjust sliders to center the red box perfectly on the hyperbola.")

### LAST MINUTE CHECKLIST:
# 1. Did you use order='F' in your Colab training flattening? 
# 2. Is '1' actually Cavity in your Excel? If Brick is row 1-30, change labels to {1: "Brick", ...}
