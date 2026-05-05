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
st.set_page_config(page_title="GPR Target Classifier", layout="centered")
st.title("📡 GPR BEMD-SVM Classifier")
st.write("Upload a raw **.rd3** file to identify the buried object.")

# 3. File Upload
uploaded_file = st.file_uploader("Choose a GPR (.rd3) file", type=["rd3"])

if uploaded_file:
    # Read binary data
    raw_data = np.frombuffer(uploaded_file.read(), dtype=np.int16)
    
    # Process 12,000 features (Matches your training)
    if len(raw_data) >= 12000:
        features = raw_data[:12000].reshape(1, -1)
        
        # Scaling & Prediction
        scaled_feat = scaler.transform(features)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        # Results
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        
        st.divider()
        st.success(f"### Result: {labels[prediction]}")
        st.write(f"**Confidence Score:** {max(prob)*100:.2f}%")
        
        # Visualization
        st.subheader("Signal IMF Visualization")
        fig, ax = plt.subplots()
        ax.imshow(features.reshape(100, 120), cmap='jet')
        st.pyplot(fig)
    else:
        st.error("File size is too small for analysis.")
