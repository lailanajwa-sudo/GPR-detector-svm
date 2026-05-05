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

st.title("📡 GPR Classification System")

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
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Create Matrix (order='F' matches MATLAB)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            # Background Removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # 3. EXTRACTION & RESIZING
            # Ensure ROI is within data bounds
            y1, x1 = min(v_start, samples_val-5), min(h_start, num_traces-5)
            y2, x2 = min(y1+box_h, samples_val), min(x1+box_w, num_traces)
            roi_raw = matrix_clean[y1:y2, x1:x2]
            
            if roi_raw.size > 0:
                # Resize to 100x120 using PIL (safe alternative to cv2)
                img = Image.fromarray(roi_raw)
                img_res = img.resize((120, 100), resample=Image.BILINEAR)
                roi_resized = np.array(img_res)
                
                # --- RANGE SCALING FIX ---
                # Your Excel data is roughly in the range [-0.035, 0.035].
                # We map the live ROI to this exact range.
                roi_min, roi_max = np.min(roi_resized), np.max(roi_resized)
                if roi_max - roi_min > 0:
                    # Scale to [-1, 1] then multiply by 0.035
                    roi_final = (((roi_resized - roi_min) / (roi_max - roi_min)) * 2 - 1) * 0.035
                else:
                    roi_final = roi_resized

                # 4. PREDICTION
                # Flatten using 'F' order (MATLAB column-major)
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
                    st.subheader("Radargram View")
                    fig, ax = plt.subplots()
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Classification Result")
                    if result == "Cavity":
                        st.success(f"### {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### {result} 🧱")
                    else:
                        st.info(f"### {result} ⚙️")

                    # Use st.pyplot to avoid the Range Error
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_final, cmap='gray', aspect='auto')
                    ax2.set_title("Input to SVM (Scaled)")
                    st.pyplot(fig2)
                    st.write(f"Value Range: {np.min(roi_final):.3f} to {np.max(roi_final):.3f}")
            else:
                st.error("ROI selection is invalid.")
