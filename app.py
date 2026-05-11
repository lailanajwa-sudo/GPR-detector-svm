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
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Logic Error: {e}")
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
    st.error("⚠️ AI Assets not found! Please ensure svm_model.pkl and scaler.pkl are in the GitHub folder and reboot the app.")
else:
    st.sidebar.header("Scan Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and convert to grayscale
        raw_img = Image.open(uploaded_file).convert('L')
        full_img = np.array(raw_img).astype(np.float64)
        display_img = mat2gray_python(full_img)
        
        # Sliders
        img_h, img_w = full_img.shape
        v_pos = st.sidebar.slider("Vertical Position (Depth)", 0, max(0, img_h - 100), 0)
        h_pos = st.sidebar.slider("Horizontal Position (Trace)", 0, max(0, img_w - 120), 0)
        
        # --- 3. FEATURE EXTRACTION ---
        roi = full_img[v_pos:v_pos+100, h_pos:h_pos+120]
        imf_cleaned = detrend(detrend(roi, axis=0), axis=1)
        features = imf_cleaned.flatten().reshape(1, -1)
        
        # --- 4. CLASSIFICATION ---
        if features.shape[1] == 12000:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            results_map = {
                1: ("CAVITY (VOID) ✅", "#238636"),
                2: ("BRICK / CONCRETE 🧱", "#d29922"),
                3: ("METAL PIPE ⚙️", "#da3633")
            }
            res_text, color = results_map.get(prediction, ("UNKNOWN ❓", "#484f58"))

            # --- 5. DISPLAY RESULTS (CROPPED FOCUS) ---
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Remove all margins and padding for a "Cropped" look
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.imshow(display_img, cmap='gray', aspect='auto')
                
                # Green bounding box
                rect = patches.Rectangle((h_pos, v_pos), 120, 100, linewidth=2, edgecolor='#00ff00', fill=False)
                ax.add_patch(rect)
                
                # CROP LOGIC: Hide axes, ticks, and labels
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown(f'''
                    <div style="padding:25px; border-radius:15px; background-color:{color}; 
                    color:white; text-align:center; font-size:24px; font-weight:bold;">
                        {res_text}
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.write("---")
                st.image(mat2gray_python(imf_cleaned), caption="12,000 BEMD Filtered Features")
                st.info("The system analyzes the pattern inside the green box to identify the material.")
        else:
            st.error(f"Feature mismatch! Expected 12000, got {features.shape[1]}.")
