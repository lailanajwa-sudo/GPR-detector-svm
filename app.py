import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD2D  # For 2D BEMD/EMD
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="GPR BEMD-SVM Classifier", layout="wide")

# --- 1. Load Data & Train Model ---
@st.cache_resource
def initial_training():
    # Load your features from the CSV in the same folder
    df = pd.read_csv('gpr_bemd.xlsx - Sheet1.csv', header=None)
    X = df.values
    # Based on your data: 30 Cavity, 30 Concrete, 30 Metal Pipe
    y = np.array([1]*30 + [2]*30 + [3]*30)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Matching your MATLAB parameters: C=1e6, Gaussian (RBF), kerneloption=12
    # gamma = 1 / (2 * sigma^2)
    gamma_val = 1 / (2 * (12**2))
    model = SVC(C=1e6, kernel='rbf', gamma=gamma_val, probability=True)
    model.fit(X_scaled, y)
    
    return model, scaler

with st.spinner("Training SVM Model..."):
    model, scaler = initial_training()

# --- 2. UI Layout ---
st.title("📡 GPR BEMD-SVM Classification System")
st.markdown("Upload a raw **.rd3** file to process and classify automatically.")

class_names = ['Cavity', 'Concrete', 'Metal Pipe']

# Sidebar
st.sidebar.header("Options")
show_plots = st.sidebar.checkbox("Show Signal Plots", value=True)

# File Uploader
uploaded_file = st.file_uploader("Upload Raw GPR (.rd3) File", type=["rd3"])

if uploaded_file is not None:
    # 3. Read Binary Data
    file_bytes = uploaded_file.read()
    # Assume 16-bit signed integer (MALA standard)
    raw_signal = np.frombuffer(file_bytes, dtype=np.int16)
    
    # 4. Processing (Placeholder for BEMD logic)
    # Your MATLAB code resized data to 100x120 = 12000 features
    if len(raw_signal) >= 12000:
        feature_vec = raw_signal[:12000].reshape(1, -1)
        
        st.subheader("Processing Results")
        col1, col2 = st.columns(2)
        
        if show_plots:
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(raw_signal[:1000])
                ax1.set_title("Raw Signal Segment")
                st.pyplot(fig1)
            
            with col2:
                # Visualization of the 2D feature matrix (100x120)
                fig2, ax2 = plt.subplots()
                ax2.imshow(feature_vec.reshape(100, 120), cmap='jet')
                ax2.set_title("Processed BEMD IMF-1 Map")
                st.pyplot(fig2)

        # 5. Classification
        scaled_feat = scaler.transform(feature_vec)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        st.divider()
        st.header(f"Result: **{class_names[prediction-1]}**")
        st.write(f"Confidence: {max(prob)*100:.2f}%")

        # 6. Save/Download
        st.subheader("💾 Save Results")
        report = f"GPR Analysis Report\nFile: {uploaded_file.name}\nResult: {class_names[prediction-1]}\nConfidence: {max(prob)*100:.2f}%"
        st.download_button("Download Report (.txt)", report, file_name="GPR_Report.txt")

    else:
        st.error("File data is too small for the 12,000 feature requirement.")
