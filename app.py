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
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except: return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

# --- 2. AUTO SCAN WITH PROBABILITY FILTER ---
def run_clean_detection(full_img, model, scaler, confidence_threshold=0.85):
    h, w = full_img.shape
    roi_h, roi_w = 100, 120
    stride = 50 # Increased stride to reduce overlapping boxes
    
    detections = []
    
    # Pre-standardize the image to help the SVM see "signals" instead of pixels
    img_std = (full_img - np.mean(full_img)) / (np.std(full_img) + 1e-7)

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            window = img_std[y:y+roi_h, x:x+roi_w]
            clean_window = detrend(detrend(window, axis=0), axis=1)
            
            # Match 11,999 features from your SVM.ipynb
            features = clean_window.flatten()[:11999].reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Get probabilities for each class
            probs = model.predict_proba(features_scaled)[0] 
            max_prob = np.max(probs)
            pred_class = np.argmax(probs) + 1 # Classes are 1, 2, 3
            
            # ONLY add detection if confidence is high
            if max_prob >= confidence_threshold:
                detections.append({
                    'x': x, 'y': y, 
                    'class': pred_class, 
                    'conf': max_prob
                })
                
    return detections

# --- 3. UI ---
st.title("📡 GPR-X Precise Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram", type=["jpg", "jpeg", "png"])
conf_level = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.85)

if uploaded_file and model:
    raw_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(raw_img).astype(np.float64)
    display_img = mat2gray_python(img_array)

    if st.sidebar.button("🔍 Run Precise Scan"):
        found = run_clean_detection(img_array, model, scaler, conf_level)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        class_info = {
            1: ("Cavity", "blue"), # Matching your blue box request
            2: ("Brick", "white"), 
            3: ("Metal", "cyan")
        }

        for d in found:
            name, color = class_info[d['class']]
            rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
            ax.add_patch(rect)
            # Label with Confidence score like your 2nd picture
            ax.text(d['x'], d['y']-5, f"{name} {d['conf']:.2f}", color=color, weight='bold', fontsize=9)

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig)
