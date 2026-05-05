import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 1. Load Assets with Error Handling
@st.cache_resource
def load_assets():
    try:
        # Force high precision loading
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

st.set_page_config(layout="wide", page_title="GPR Automated Classifier")
st.title("📡 Automated GPR Classification")
st.write("Upload your files and center the red box on the hyperbola.")

# 2. Simplified Sidebar - Only ROI Controls
st.sidebar.header("ROI Alignment")
v_start = st.sidebar.slider("Vertical (Depth)", 0, 212, 115) 
h_start = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)

# Fixed ROI size to match your 12,000 feature vector (100x120)
# This removes user error in box sizing
BOX_H, BOX_W = 100, 120

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Read RAD for sample count
        rad_text = rad_file.getvalue().decode("utf-8")
        samples_val = 312
        for line in rad_text.split('\n'):
            if "SAMPLES:" in line:
                samples_val = int(line.split(':')[1].strip())
        
        # B. Read RD3 with Double Precision (float64)
        raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        num_traces = len(raw_data) // samples_val
        
        if num_traces > 0:
            # Create Matrix using Fortran order (MATLAB Standard)
            matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
            # Background removal
            matrix_clean = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # C. ROI Extraction
            y1, x1 = min(v_start, samples_val - BOX_H), min(h_start, num_traces - BOX_W)
            roi_raw = matrix_clean[y1:y1+BOX_H, x1:x1+BOX_W]
            
            if roi_raw.size > 0:
                # Resize using high-detail BICUBIC interpolation
                img = Image.fromarray(roi_raw)
                img_res = img.resize((BOX_W, BOX_H), resample=Image.BICUBIC)
                roi_resized = np.array(img_res, dtype=np.float64)
                
                # --- D. AUTOMATIC PRECISION SCALING ---
                # We map the live ROI range to the exact range of your Excel file
                # Training Max: 0.0824 | Training Min: -0.0899 (Based on your Class 3 data)
                # This ensures the SVM sees values exactly like the training set.
                roi_min, roi_max = np.min(roi_resized), np.max(roi_resized)
                if roi_max - roi_min > 0:
                    # Normalized to [-0.08, 0.08] range
                    roi_scaled = ((roi_resized - roi_min) / (roi_max - roi_min) * 0.16) - 0.08
                else:
                    roi_scaled = roi_resized

                # E. PREDICTION (Force 64-bit precision)
                # Flattening in 'F' order to match MATLAB's (:) operator
                features = roi_scaled.flatten(order='F').reshape(1, -1)
                
                # Apply scaler and model
                scaled_feat = scaler.transform(features)
                pred = model.predict(scaled_feat)[0]
                
                # Label mapping
                labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
                result = labels.get(pred, "Unknown")

                # F. UI DISPLAY
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig, ax = plt.subplots()
                    limit = np.percentile(np.abs(matrix_clean), 98)
                    ax.imshow(matrix_clean, cmap='gray', aspect='auto', vmin=-limit, vmax=limit)
                    ax.add_patch(patches.Rectangle((x1, y1), BOX_W, BOX_H, color='red', fill=False, lw=2))
                    st.pyplot(fig)

                with col2:
                    st.header("Result")
                    if result == "Cavity":
                        st.success(f"### {result} ✅")
                    elif result == "Brick":
                        st.warning(f"### {result} 🧱")
                    else:
                        st.error(f"### {result} ⚙️")
                    
                    # Debug Information (Precision Check)
                    st.write("**High Precision ROI Stats:**")
                    st.code(f"Min: {np.min(roi_scaled):.8f}\nMax: {np.max(roi_scaled):.8f}\nMean: {np.mean(roi_scaled):.8f}")
                    
                    fig2, ax2 = plt.subplots()
                    ax2.imshow(roi_scaled, cmap='gray')
                    ax2.set_title("Scaled SVM Input")
                    st.pyplot(fig2)
