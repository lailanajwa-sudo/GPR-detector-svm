import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import io

# Page Config
st.set_page_config(page_title="GPR Automated Classifier", layout="wide")

# 1. Load Data & Train SVM
@st.cache_resource
def train_model():
    # Load your features
    df = pd.read_csv('gpr_bemd.xlsx - Sheet1.csv', header=None)
    X = df.values
    # Labeling: 30 Cavity (1), 30 Concrete (2), 30 Metal (3)
    y = np.array([1]*30 + [2]*30 + [3]*30)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVM with your MATLAB parameters: C=1e6, Gamma based on kerneloption 12
    gamma_val = 1 / (2 * (12**2))
    model = SVC(C=1e6, kernel='rbf', gamma=gamma_val, probability=True)
    model.fit(X_scaled, y)
    
    return model, scaler

with st.spinner("System initializing... Training SVM model."):
    model, scaler = train_model()

# 2. UI Layout
st.title("📡 GPR Classification System")
st.markdown("Automated detection of **Cavity, Concrete, and Metal Pipe** using BEMD-SVM.")

class_names = ['Cavity', 'Concrete', 'Metal Pipe']

# File Uploader
uploaded_file = st.file_uploader("Upload Raw GPR Data (.rd3)", type=["rd3"])

if uploaded_file is not None:
    # Read binary
    file_bytes = uploaded_file.read()
    raw_data = np.frombuffer(file_bytes, dtype=np.int16)
    
    # Process only if data is enough (12000 features)
    if len(raw_data) >= 12000:
        feature_vec = raw_data[:12000].reshape(1, -1)
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visualized Signal")
            fig, ax = plt.subplots()
            ax.imshow(feature_vec.reshape(100, 120), cmap='jet')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Classification Result")
            scaled_feat = scaler.transform(feature_vec)
            prediction = model.predict(scaled_feat)[0]
            conf = model.predict_proba(scaled_feat)[0]
            
            res = class_names[prediction-1]
            st.success(f"### Detected: **{res}**")
            st.write(f"Confidence Level: {max(conf)*100:.2f}%")
            
            # Save/Download Option
            st.markdown("---")
            report = f"Analysis Report\nResult: {res}\nConfidence: {max(conf)*100:.2f}%"
            st.download_button("Download Report (.txt)", report, file_name="GPR_Result.txt")
    else:
        st.error("Data too small. Requires at least 12,000 points.")
