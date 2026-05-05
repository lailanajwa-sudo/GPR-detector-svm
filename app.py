import streamlit as st
import numpy as np
import joblib
from PyEMD import BEMD  # You must add 'PyEMD' to requirements.txt
from PIL import Image

# ... (Loading assets same as before) ...

def extract_bemd_features(roi_data):
    """
    Decomposes the 100x120 ROI into IMFs and returns the feature vector.
    """
    # Initialize BEMD
    bemd = BEMD()
    
    # Decompose the 2D ROI
    # This produces a 3D array: [Number_of_IMFs, 100, 120]
    IMFs = bemd.bemd(roi_data)
    
    # In most GPR research, IMF1 or IMF2 contains the target 'shape'
    # We take IMF1 as the primary feature
    imf1 = IMFs[0] 
    
    # Flatten to match your 12,000 feature vector
    # Use 'F' order if your training was in MATLAB
    return imf1.flatten(order='F').reshape(1, -1)

# ... (Inside the file processing loop) ...

            if roi_raw.size > 0:
                # 1. Standardize and Resize
                img = Image.fromarray(roi_raw).resize((120, 100), Image.BICUBIC)
                roi_resized = np.array(img, dtype=np.float64)
                
                # 2. EXTRACT BEMD FEATURES (The Missing Step)
                st.write("🔄 Extracting BEMD Features...")
                try:
                    bemd_features = extract_bemd_features(roi_resized)
                    
                    # 3. PREDICT
                    scaled_feat = scaler.transform(bemd_features)
                    pred = model.predict(scaled_feat)[0]
                    
                    # ... (Display results) ...
                except Exception as e:
                    st.error(f"BEMD Error: {e}. Check if ROI contains enough detail.")
