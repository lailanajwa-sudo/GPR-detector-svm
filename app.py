import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. BEMD FEATURE SIMULATION ---
def get_bemd_features(roi_matrix):
    """
    Simulates the BEMD IMF extraction.
    Your Excel data (gpr_bemd) is derived from IMFs which act as 
    high-pass filters to highlight the hyperbola edges.
    """
    # 1. Row-wise and Column-wise detrending (simulates BEMD IMF1)
    detrended = roi_matrix - np.mean(roi_matrix, axis=0)
    detrended = detrended - np.mean(detrended, axis=1, keepdims=True)
    
    # 2. Match the exact 6-decimal precision and range of your CSV
    # Training data max is ~0.08. We scale the ROI to this limit.
    std_target = 0.0055 # Average std of Class 1 (Cavity) in your Excel
    current_std = np.std(detrended)
    
    if current_std > 0:
        features_fixed = (detrended / current_std) * std_target
    else:
        features_fixed = detrended
        
    # 3. Flatten using 'F' (Fortran/MATLAB) order to match gpr_bemd.xlsx
    return features_fixed.flatten(order='F').reshape(1, -1)

# --- 3. UI SETUP ---
st.set_page_config(layout="wide")
st.title("📡 GPR BEMD-SVM Classifier")

st.sidebar.header("ROI Adjustment")
v_pos = st.sidebar.slider("Vertical (Depth)", 0, 212, 115)
h_pos = st.sidebar.slider("Horizontal (Trace)", 0, 450, 210)

uploaded_files = st.file_uploader("Upload .rad and .rd3", type=["rad", "rd3"], accept_multiple_files=True)

if len(uploaded_files) == 2 and model is not None:
    rad_file = next((f for f in uploaded_files if f.name.endswith('.rad')), None)
    rd3_file = next((f for f in uploaded_files if f.name.endswith('.rd3')), None)

    if rad_file and rd3_file:
        # A. Parse RAD
        samples = 312
        content = rad_file.getvalue().decode("utf-8")
        for line in content.split('\n'):
            if "SAMPLES:" in line:
                samples = int(line.split(':')[1].strip())
        
        # B. Read Binary
        raw = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(np.float64)
        traces = len(raw) // samples
        
        if traces > 0:
            matrix = raw[:samples*traces].reshape((samples, traces), order='F')
            # Background removal
            bg_rem = matrix - np.mean(matrix, axis=1, keepdims=True)
            
            # C. Extract 100x120 ROI
            y1, x1 = min(v_pos, samples-100), min(h_pos, traces-120)
            roi = bg_rem[y1:y1+100, x1:x1+120]
            
            # D. Feature Extraction & Prediction
            # Ensure ROI is resized to exactly 100x120
            img = Image.fromarray(roi).resize((120, 100), Image.BICUBIC)
            roi_ready = np.array(img)
            
            # Extract BEMD-style features
            final_features = get_bemd_features(roi_ready)
            
            # SVM Prediction
            scaled_input = scaler.transform(final_features)
            prediction = model.predict(scaled_input)[0]
            
            labels = {1: "Cavity", 2: "Brick", 3: "Metal Pipe"}
            result = labels.get(prediction, "Unknown")
            
            # E. Display Results
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots()
                lim = np.percentile(np.abs(bg_rem), 98)
                ax.imshow(bg_rem, cmap='gray', aspect='auto', vmin=-lim, vmax=lim)
                ax.add_patch(patches.Rectangle((x1, y1), 120, 100, color='red', fill=False, lw=2))
                st.pyplot(fig)
            
            with col2:
                st.header("Classification")
                if result == "Cavity":
                    st.success(f"### {result} ✅")
                elif result == "Brick":
                    st.warning(f"### {result} 🧱")
                else:
                    st.error(f"### {result} ⚙️")
                
                st.write("**BEMD Feature Stats (6 Decimals):**")
                st.code(f"Max: {np.max(final_features):.6f}\nMin: {np.min(final_features):.6f}")
                
                fig2, ax2 = plt.subplots()
                ax2.imshow(roi_ready, cmap='gray')
                ax2.set_title("Input to BEMD Processor")
                st.pyplot(fig2)
