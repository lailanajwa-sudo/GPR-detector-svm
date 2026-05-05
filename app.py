import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2  # For resizing logic

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="GPR Assignment Final", layout="wide")
st.title("📡 MALA GPR Classification System")

# Sidebar for ROI Positioning
st.sidebar.header("ROI Selection")
# Let the user define the box size to calculate the scale
roi_w = st.sidebar.slider("ROI Width (col)", 50, 300, 120)
roi_h = st.sidebar.slider("ROI Height (row)", 50, 200, 100)
h_start = st.sidebar.slider("Horizontal Position", 0, 350, 200)
v_start = st.sidebar.slider("Vertical Position", 0, 212, 120)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # 1. Parse RAD for samples count
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # 2. Read Binary Data
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # 3. Process Matrix
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 4. ROI EXTRACTION & RESIZING (Your scale logic)
            # We crop the user-defined box and resize it to 100x120
            roi_raw = matrix_clean[v_start : v_start+roi_h, h_start : h_start+roi_w]
            
            if roi_raw.size > 0:
                # Resize to 100 samples (height) and 120 traces (width)
                # This matches your scalex=(120/col) logic
                roi_resized = cv2.resize(roi_raw, (120, 100), interpolation=cv2.INTER_LINEAR)
                
                # --- STATISTICAL NORMALIZATION ---
                # Your training data has mean ~0 and std ~0.005. 
                # We force the live data to match this scale.
                if np.std(roi_resized) > 0:
                    roi_norm = (roi_resized - np.mean(roi_resized)) / np.std(roi_resized)
                    roi_final = roi_norm * 0.005  # Force std to match training data
                else:
                    roi_final = roi_resized

                # 5. UI Layout
                col1, col2 = st.columns([2, 1])
                limit = np.percentile(np.abs(matrix_clean), 98)

                with col1:
                    st.subheader("Radargram")
                    fig, ax = plt.subplots()
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    rect = patches.Rectangle((h_start, v_start), roi_w, roi_h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Analysis")
                    # Prediction using Fortran flattening (order='F')
                    features = roi_final.flatten(order='F').reshape(1, -1)
                    scaled_feat = scaler.transform(features)
                    pred = model.predict(scaled_feat)[0]
                    
                    # Labels based on your Excel: 1=Cavity, 2=Brick, 3=Metal Pipe
                    labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                    result = labels.get(pred, "Unknown")
                    
                    if result == "Cavity":
                        st.success(f"### Result: {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### Result: {result} 🧱")
                    else:
                        st.info(f"### Result: {result} ⚙️")

                    # Preview what the SVM sees
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray', aspect='auto')
                    ax2.set_title("Resized & Normalized Input")
                    st.pyplot(fig2)
            else:
                st.error("ROI selection is out of bounds.")
