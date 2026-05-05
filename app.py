import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_assets()

st.set_page_config(layout="wide")
st.title("📡 GPR High-Precision Classification")

# 2. Sidebar for ROI Controls
st.sidebar.header("Target Selection")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 400, 210)
box_w = st.sidebar.slider("Box Width (col)", 50, 250, 120)
box_h = st.sidebar.slider("Box Height (row)", 50, 200, 100)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # Parse RAD
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # Read Binary RD3
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Create Matrix (order='F' matches MATLAB)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            # Background Removal (Mean subtraction)
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 3. EXTRACTION & RESIZING
            y1, x1 = min(v_start, samples_val-5), min(h_start, num_traces-5)
            y2, x2 = min(y1+box_h, samples_val), min(x1+box_w, num_traces)
            roi_raw = matrix_clean[y1:y2, x1:x2]
            
            if roi_raw.size > 0:
                # Resize to 100x120 using PIL
                img = Image.fromarray(roi_raw)
                img_res = img.resize((120, 100), resample=Image.BICUBIC) # High quality resize
                roi_resized = np.array(img_res)
                
                # --- HIGH PRECISION RANGE SCALING ---
                # Your Excel training data max is 0.032574
                target_limit = 0.032574
                roi_min, roi_max = np.min(roi_resized), np.max(roi_resized)
                
                if roi_max - roi_min > 0:
                    # Map to exactly [-0.032574, +0.032574] with 6 decimal precision
                    roi_final = (((roi_resized - roi_min) / (roi_max - roi_min)) * 2 - 1) * target_limit
                    roi_final = np.round(roi_final, 6) # Force 6 decimal places
                else:
                    roi_final = roi_resized

                # 4. PREDICTION
                # Flatten using 'F' order (column-major) to match the 12,000 feature vector
                features = roi_final.flatten(order='F').reshape(1, -1)
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # Labels: 1=Cavity, 2=Brick, 3=Metal Pipe
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")
                
                # 5. UI DISPLAY
                col1, col2 = st.columns([2, 1])
                limit = np.percentile(np.abs(matrix_clean), 98)

                with col1:
                    st.subheader("Radargram Selection")
                    fig, ax = plt.subplots()
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Classification")
                    if result == "Cavity":
                        st.success(f"### Detected: {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### Detected: {result} 🧱")
                    else:
                        st.info(f"### Detected: {result} ⚙️")

                    # High Precision Stats
                    st.write(f"**ROI Stats (6 Decimals):**")
                    st.code(f"Max: {np.max(roi_final):.6f}\nMin: {np.min(roi_final):.6f}\nMean: {np.mean(roi_final):.6f}")

                    # ROI Preview
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray', aspect='auto')
                    ax2.set_title("Input Sample (Resized/Normalized)")
                    st.pyplot(fig2)
            else:
                st.error("Invalid ROI Selection")
