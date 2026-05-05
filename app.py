import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load the "Best" files
@st.cache_resource
def load_assets():
    # Ensure these files are in your GitHub root folder
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="GPR Target Identification", layout="wide")
st.title("📡 GPR Target Classifier: Metal Pipe, Cavity, & Concrete")

uploaded_file = st.file_uploader("Upload .rd3 file", type=["rd3"])

if uploaded_file:
    # Read binary data
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16).astype(float)
    
    # We need exactly 12,000 points (100 samples x 120 traces)
    if len(raw_data) >= 12000:
        # Step A: Reshape correctly
        # order='F' (Fortran) is standard for GPR where data is stored Trace by Trace
        matrix = raw_data[:12000].reshape((100, 120), order='F')
        
        # Step B: Signal Processing (Standardize for SVM)
        # 1. Background removal (removes horizontal air-wave stripes)
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
        
        # 2. Trace-wise normalization (makes signal levels consistent)
        matrix = matrix / (np.max(np.abs(matrix)) + 1e-6)

        # Step C: Prepare for SVM
        # The SVM was trained on a flattened version of your excel data
        # Ensure the flatten order matches your training (usually C-style for pandas/excel)
        feature_vector = matrix.flatten().reshape(1, -1)
        
        # Step D: Prediction
        scaled_feat = scaler.transform(feature_vector)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        result = labels.get(prediction, "Unknown")

        # --- UI DISPLAY ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("B-Scan Radargram View")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Seismic colormap is best for spotting hyperbolic metal reflections
            v_limit = np.percentile(np.abs(matrix), 98) # Adjust contrast
            im = ax.imshow(matrix, cmap='seismic', aspect='auto', 
                           vmin=-v_limit, vmax=v_limit, interpolation='bilinear')
            
            ax.set_title(f"Target: {result}")
            ax.set_xlabel("Trace Number (Horizontal Distance)")
            ax.set_ylabel("Time Sample (Depth)")
            plt.colorbar(im, label="Normalized Amplitude")
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Result")
            if result == "Metal Pipe":
                st.success(f"### Target: {result}")
            else:
                st.warning(f"### Target: {result}")
                
            st.metric("Confidence Score", f"{max(prob)*100:.2f}%")
            
            st.write("""
            **How to read this radargram:**
            * **Metal Pipe:** Look for a very bright, sharp 'V' or 'U' shape.
            * **Cavity:** Usually a wider, dimmer hyperbola.
            * **Concrete:** Often appears as a dense, flat, or jagged horizontal reflection.
            """)
    else:
        st.error(f"File too small. Data points: {len(raw_data)}. Need: 12,000.")
