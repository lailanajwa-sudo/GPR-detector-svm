import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from PIL import Image

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        # Loading the model and scaler trained in your SVM.ipynb
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="GPR-X Image Detector", layout="wide")
st.title("📡 GPR-X Detection (Image-based SVM)")

if model is None:
    st.error("⚠️ AI Assets (svm_model.pkl / scaler.pkl) not found! Please ensure they are in the same folder as this script.")
else:
    # Sidebar for control
    st.sidebar.header("Scan Settings")
    
    # File uploader for images instead of .rd3/.rad
    uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load image and convert to grayscale to match numerical matrix processing
        raw_img = Image.open(uploaded_file).convert('L')
        full_img = np.array(raw_img).astype(np.float64)
        
        # Normalize for visualization
        display_img = mat2gray_python(full_img)
        
        # Dynamic sliders based on uploaded image dimensions
        img_h, img_w = full_img.shape
        v_pos = st.sidebar.slider("Vertical Position (Depth)", 0, max(0, img_h - 100), 0)
        h_pos = st.sidebar.slider("Horizontal Position (Trace)", 0, max(0, img_w - 120), 0)
        
        # --- 3. FEATURE EXTRACTION ---
        # Extract the 100x120 Region of Interest (ROI)
        roi = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        
        # Apply filtering (Detrending) to mirror BEMD feature cleaning
        imf_cleaned = detrend(detrend(roi, axis=0), axis=1)
        
        # Flatten to exactly 12,000 features (100 * 120)
        features = imf_cleaned.flatten().reshape(1, -1)
        
        # --- 4. CLASSIFICATION ---
        # Ensure the feature count matches the Scaler/SVM expectation exactly
        if features.shape[1] == 12000:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            # Map labels from SVM training (1=Cavity, 2=Brick, 3=Metal)
            results_map = {
                1: ("CAVITY (VOID) ✅", "#238636"),
                2: ("BRICK / CONCRETE 🧱", "#d29922"),
                3: ("METAL PIPE ⚙️", "#da3633")
            }
            res_text, color = results_map.get(prediction, ("UNKNOWN ❓", "#484f58"))

            # --- 5. DISPLAY RESULTS ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(display_img, cmap='gray', aspect='auto')
                # Add bounding box for ROI
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                ax.add_patch(rect)
                plt.axis('off')
                st.pyplot(fig)

            with col2:
                st.markdown(f'''
                    <div style="padding:25px; border-radius:15px; background-color:{color}; 
                    color:white; text-align:center; font-size:24px; font-weight:bold;">
                        {res_text}
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.write("---")
                st.image(mat2gray_python(imf_cleaned), caption="12,000 BEMD Filtered Features (Input to SVM)")
                st.info("Analysis complete. The SVM model identifies materials based on pixel intensity patterns.")
        else:
            st.error(f"Feature mismatch! Model expects 12000 features, but ROI produced {features.shape[1]}.")
