import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load the "Brain"
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. UI Styling
st.set_page_config(page_title="GPR Radargram Classifier", layout="centered")
st.title("📡 GPR Radargram Analysis & Classification")
st.write("Upload a raw **.rd3** file to visualize the radargram and identify the target.")

# 3. File Upload
uploaded_file = st.file_uploader("Choose a GPR (.rd3) file", type=["rd3"])

if uploaded_file:
    # Read binary data
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16)
    
    # Process 12,000 features (Matches your training: 100 rows x 120 columns)
    if len(raw_data) >= 12000:
        # We take the first 12000 points to match your SVM input
        features = raw_data[:12000].reshape(1, -1)
        
        # Scaling & Prediction
        scaled_feat = scaler.transform(features)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        # Labels
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        
        st.divider()
        
        # Display Result
        st.success(f"### Predicted Target: **{labels[prediction]}**")
        st.write(f"**Confidence Level:** {max(prob)*100:.2f}%")
        
        # 4. RADARGRAM VISUALIZATION
        st.subheader("GPR Radargram (B-Scan)")
        
        # Reshape data to 2D matrix (100 samples x 120 traces)
        # Note: GPR data is usually displayed with Depth/Time on the Y-axis
        radargram = features.reshape(100, 120)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 'gray' makes it look like a classic radargram
        # 'seismic' is also good for highlighting wave peaks/troughs
        im = ax.imshow(radargram, cmap='gray', aspect='auto', interpolation='bilinear')
        
        ax.set_title(f"Radargram View: {uploaded_file.name}")
        ax.set_xlabel("Traces (Horizontal Distance)")
        ax.set_ylabel("Samples (Depth/Time)")
        plt.colorbar(im, label='Amplitude')
        
        st.pyplot(fig)
        
    else:
        st.error(f"File size too small. Found {len(raw_data)} points, but need at least 12,000.")
