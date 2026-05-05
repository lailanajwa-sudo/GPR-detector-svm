import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("📡 GPR Classification - Final Fix")

# 2. Sidebar Controls
st.sidebar.header("ROI Alignment")
# Based on your screenshots, these are common hyperbola areas
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 400, 210)
# Manual scaling factors mentioned in your prompt
box_w = st.sidebar.slider("Box Width", 50, 200, 120)
box_h = st.sidebar.slider("Box Height", 50, 200, 100)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # Parse RAD
        rad_content = rad_file.getvalue().decode("utf-8")
        samples_val = 312 
        for line in rad_content.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # Read Binary
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Create Matrix (order='F' is critical for MATLAB data)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            
            # Background Removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 3. EXTRACTION & RESIZING (PIL avoids the cv2 error)
            roi_raw = matrix_clean[v_start : v_start+box_h, h_start : h_start+box_w]
            
            if roi_raw.size > 0:
                # Resize to exactly 100x120 (Matches your BEMD 12000 feature vector)
                img = Image.fromarray(roi_raw)
                img_res = img.resize((120, 100), resample=Image.BILINEAR)
                roi_resized = np.array(img_res)
                
                # --- THE CRITICAL NORMALIZATION FIX ---
                # We force the ROI to have a Standard Deviation of 0.005 
                # to match your Excel's 'gpr_bemd' scale perfectly.
                roi_std = np.std(roi_resized)
                if roi_std > 0:
                    # Step A: Z-score (makes mean 0, std 1)
                    roi_norm = (roi_resized - np.mean(roi_resized)) / roi_std
                    # Step B: Scale down to the "Excel Range" (std 0.005)
                    roi_final = roi_norm * 0.005
                else:
                    roi_final = roi_resized

                # 4. PREDICTION
                # Flatten using 'F' to match MATLAB's bemd_gpr.m sequence
                features = roi_final.flatten(order='F').reshape(1, -1)
                
                # Apply the original training scaler
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # LABEL MAPPING (1:Cavity, 2:Brick, 3:Metal)
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")
                
                # 5. UI DISPLAY
                col1, col2 = st.columns([2, 1])
                limit = np.percentile(np.abs(matrix_clean), 98)

                with col1:
                    fig, ax = plt.subplots()
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    rect = patches.Rectangle((h_start, v_start), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    st.pyplot(fig)

                with col2:
                    st.subheader(f"Prediction: {result}")
                    if result == "Cavity":
                        st.success("Target Identified! ✅")
                    elif result == "Brick":
                        st.warning("Target: Brick 🧱")
                    else:
                        st.info("Target: Metal Pipe ⚙️")

                    # Debug: Show the ROI "Value Range"
                    st.write(f"ROI Max Value: {np.max(roi_final):.4f}")
                    st.image(roi_final, caption="Normalized Input", use_container_width=True)
            else:
                st.error("ROI is out of bounds. Adjust sliders.")
