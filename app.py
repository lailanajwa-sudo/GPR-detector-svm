import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load the Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- UI Setup ---
st.set_page_config(page_title="Mala GPR Analyzer", layout="wide")
st.title("📡 Mala RD3 Radargram & Target Classifier")

uploaded_file = st.file_uploader("Upload Mala .rd3 File", type=["rd3"])

if uploaded_file:
    # 2. Read Data (Equivalent to fread(fid, [samples, inf], 'int16'))
    raw_bytes = uploaded_file.read()
    raw_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(float)
    
    samples_per_trace = 100 # From your training configuration
    
    if len(raw_data) >= 12000:
        # 3. Reshape Trace-by-Trace (Fortran order 'F' matches MATLAB's [samples, inf])
        # This creates a matrix where each column is one radar shot (trace)
        matrix = raw_data[:12000].reshape((samples_per_trace, 120), order='F')
        
        # 4. Correct Orientation (Equivalent to flipud(data) in your MATLAB code)
        # This ensures the "Surface" is at the top (0 ns)
        matrix_flipped = np.flipud(matrix)

        # 5. Signal Enhancement (Background Removal)
        # Subtracting the horizontal mean removes the 'ringing' noise
        processed_matrix = matrix_flipped - np.mean(matrix_flipped, axis=1, keepdims=True)

        # 6. SVM Classification
        # We flatten the processed matrix to match the SVM input format
        feature_vector = processed_matrix.flatten().reshape(1, -1)
        scaled_feat = scaler.transform(feature_vector)
        prediction = model.predict(scaled_feat)[0]
        prob = model.predict_proba(scaled_feat)[0]
        
        labels = {1: "Cavity", 2: "Concrete", 3: "Metal Pipe"}
        result = labels.get(prediction, "Unknown")

        # --- Display Section ---
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Radargram (B-Scan)")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Using 'gray' to match your MATLAB code's colormap
            # vmin/vmax tightens the contrast to make hyperbolas pop
            limit = np.percentile(np.abs(processed_matrix), 98)
            im = ax.imshow(processed_matrix, cmap='gray', aspect='auto', 
                           vmin=-limit, vmax=limit, interpolation='bilinear')
            
            ax.set_ylabel("Time Samples (Depth)")
            ax.set_xlabel("Trace Number")
            plt.colorbar(im, label="Amplitude")
            st.pyplot(fig)

        with col2:
            st.subheader("Analysis")
            st.metric("Detected Target", result)
            st.metric("Confidence", f"{max(prob)*100:.2f}%")
            
            st.write("**Mala Format Logic Applied:**")
            st.write("- Trace-wise reshaping")
            st.write("- Vertical Flip (`flipud`)")
            st.write("- Background Removal")
            
    else:
        st.error(f"Incomplete data. Found {len(raw_data)} samples, need 12,000.")
