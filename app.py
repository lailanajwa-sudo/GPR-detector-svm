import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load the Pre-trained Assets
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- Page Layout ---
st.set_page_config(page_title="GPR Hyperbolic Analysis", layout="wide")
st.title("📡 GPR Hyperbolic Pattern Analysis & Classification")
st.markdown("---")

# 2. Sidebar for Image Settings
st.sidebar.header("Image Processing Settings")
cmap_option = st.sidebar.selectbox("Color Map", ["gray", "seismic", "RdBu", "bone"])
gain_val = st.sidebar.slider("Signal Gain", 1.0, 10.0, 2.0)

# 3. File Upload
uploaded_file = st.file_uploader("Upload Raw GPR (.rd3) File", type=["rd3"])

if uploaded_file:
    # Read binary data (16-bit signed integer)
    raw_bytes = uploaded_file.read()
    raw_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(float)
    
    # Requirement Check: 100 samples x 120 traces = 12,000 points
    if len(raw_data) >= 12000:
        # a. Extract and Reshape (100 rows x 120 columns)
        # Rows = Time/Depth, Columns = Distance/Traces
        matrix = raw_data[:12000].reshape(100, 120)
        
        # b. Signal Processing for better Hyperbola visibility
        # 1. DC Removal (Subtract mean of each trace to remove horizontal noise)
        matrix = matrix - np.mean(matrix, axis=0)
        
        # 2. Apply Gain
        matrix = matrix * gain_val
        
        # c. Classification (Using the 1D feature vector for SVM)
        feature_vector = matrix.flatten().reshape(1, -1)
        scaled_feat = scaler.transform(feature_vector)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        
        # --- Display Results ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("B-Scan Radargram")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Displaying the radargram
            # vmin/vmax helps in contrast for hyperbolic patterns
            v_limit = np.percentile(np.abs(matrix), 95) 
            im = ax.imshow(matrix, 
                           cmap=cmap_option, 
                           aspect='auto', 
                           interpolation='bilinear',
                           vmin=-v_limit, vmax=v_limit)
            
            ax.set_xlabel("Traces (Position)")
            ax.set_ylabel("Samples (Time/Depth)")
            plt.colorbar(im, label='Amplitude')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Classification")
            st.metric("Detected Target", labels[prediction])
            st.metric("Confidence Score", f"{max(prob)*100:.2f}%")
            
            st.info("""
            **Hyperbolic Interpretation:**
            The inverted 'U' shapes in the image represent reflections from point targets. 
            The peak of the hyperbola indicates the exact location of the object.
            """)
            
            # Download Option
            report = f"Target: {labels[prediction]}\nConfidence: {max(prob)*100:.2f}%"
            st.download_button("📩 Download Results", report, file_name="GPR_Analysis.txt")

    else:
        st.error(f"File size error. Need 12,000 points, but file has {len(raw_data)}.")
