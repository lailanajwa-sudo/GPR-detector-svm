import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load trained SVM and Scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}. Ensure svm_model.pkl and scaler.pkl are in the repository.")
        return None, None

model, scaler = load_assets()

# Page configuration
st.set_page_config(layout="wide", page_title="GPR High-Precision Classifier")
st.title("📡 GPR Classification System (BEMD Optimized)")

# 2. Sidebar - ROI and Sensitivity Tuning
st.sidebar.header("🎯 Target Selection")
v_start = st.sidebar.slider("Vertical (Depth) Position", 0, 250, 115) 
h_start = st.sidebar.slider("Horizontal (Trace) Position", 0, 450, 210)

st.sidebar.header("📏 ROI Size")
box_w = st.sidebar.slider("ROI Width", 50, 200, 120)
box_h = st.sidebar.slider("ROI Height", 50, 200, 100)

st.sidebar.header("⚖️ SVM Sensitivity")
# Your Excel data (gpr_bemd) has very specific standard deviations:
# Cavity ~0.005 | Brick ~0.006 | Metal ~0.015
target_std = st.sidebar.slider("Signal Intensity (Std Dev)", 0.001000, 0.030000, 0.005500, format="%.6f")
st.sidebar.info("Lower Intensity (0.003-0.006) helps detect Cavities. Higher (0.015+) usually triggers Metal.")

# File Uploader
uploaded_files = st.file_uploader("Upload MALA .rad and .rd3 files", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # --- A. PARSE RAD FILE ---
        rad_text = rad_file.getvalue().decode("utf-8")
        samples_val = 312 # Default
        for line in rad_text.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # --- B. PROCESS RD3 BINARY DATA ---
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Reshape using 'F' (Fortran/MATLAB order)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            # Background Removal (Average subtraction)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # --- C. EXTRACT & NORMALIZE ROI ---
            # Ensure indices stay within matrix bounds
            y1, x1 = min(v_start, samples_val - 10), min(h_start, num_traces - 10)
            y2, x2 = min(y1 + box_h, samples_val), min(x1 + box_w, num_traces)
            roi_raw = matrix_clean[y1:y2, x1:x2]
            
            if roi_raw.size > 0:
                # 1. Resize to 100x120 using BICUBIC for high detail
                img = Image.fromarray(roi_raw)
                img_res = img.resize((120, 100), resample=Image.BICUBIC)
                roi_resized = np.array(img_res)
                
                # 2. Match Statistical Scale (The "Metal Pipe Fix")
                # We normalize the live ROI to have the exact Std Dev selected in the sidebar
                current_std = np.std(roi_resized)
                if current_std > 0:
                    roi_norm = (roi_resized - np.mean(roi_resized)) / current_std
                    roi_final = np.round(roi_norm * target_std, 6) # Force 6 decimals
                else:
                    roi_final = roi_resized

                # --- D. SVM PREDICTION ---
                # Flatten using 'F' order to match MATLAB vectorization (12000 features)
                features = roi_final.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                prediction = model.predict(scaled_feat)[0]
                
                # Labels: 1=Cavity, 2=Brick, 3=Metal Pipe
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(prediction, "Unknown")

                # --- E. UI RESULTS DISPLAY ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Radargram Visualization")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    limit = np.percentile(np.abs(matrix_clean), 98)
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    # Draw the ROI box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.set_title(f"Target Selection: {x2-x1} traces x {y2-y1} samples")
                    st.pyplot(fig)

                with col2:
                    st.subheader("Classification Outcome")
                    if result == "Cavity":
                        st.success(f"### Result: {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### Result: {result} 🧱")
                    else:
                        st.info(f"### Result: {result} ⚙️")
                    
                    # Display metrics to debug "Metal Pipe" issue
                    st.write("**Signal Metrics (6 Decimals):**")
                    st.code(f"Target Std: {target_std:.6f}\nROI Mean:   {np.mean(roi_final):.6f}\nROI Max:    {np.max(roi_final):.6f}")

                    # ROI Preview (What the SVM sees)
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray', aspect='auto')
                    ax2.set_title("Input Sample (Resized & Normalized)")
                    st.pyplot(fig2)
            else:
                st.error("Invalid ROI selection area.")
        else:
            st.error("RD3 file appears empty or corrupted.")
    else:
        st.error("Please upload both .rad and .rd3 files simultaneously.")
