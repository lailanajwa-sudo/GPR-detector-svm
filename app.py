import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import time

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load the model and scaler you trained in Colab
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING FUNCTIONS (Mirroring MATLAB) ---
def preprocess_roi(roi_img):
    """
    Standardizes the ROI to match the MATLAB BEMD extraction logic.
    """
    # Convert to grayscale and resize to 100x120
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray equivalent
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Histogram Equalization (histeq) equivalent to boost hyperbola contrast
    # Using a simple cumulative distribution function (CDF) mapping
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    
    # Flatten to 12,000 features
    return img_eq.flatten().reshape(1, -1)

# --- 3. DETECTION ENGINE ---
def run_detection(full_img, threshold=0.90):
    w, h = full_img.size
    roi_w, roi_h = 120, 100
    stride = 40  # Moves the window by 40 pixels each time
    
    detections = []
    
    # Loading indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total steps for the progress bar
    x_steps = range(0, w - roi_w, stride)
    y_steps = range(0, h - roi_h, stride)
    total_steps = len(x_steps) * len(y_steps)
    current_step = 0

    for y in y_steps:
        for x in x_steps:
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            status_text.text(f"Analyzing scan area... {int((current_step/total_steps)*100)}%")
            
            # Extract ROI
            roi = full_img.crop((x, y, x + roi_w, y + roi_h))
            features = preprocess_roi(roi)
            
            # Predict
            features_scaled = scaler.transform(features)
            probs = model.predict_proba(features_scaled)[0]
            best_class = np.argmax(probs)
            confidence = probs[best_class]
            
            # Only record if it's not "background" (High confidence filter)
            if confidence >= threshold:
                detections.append({
                    'box': [x, y, roi_w, roi_h],
                    'class': best_class + 1,
                    'conf': confidence
                })
                
    progress_bar.empty()
    status_text.empty()
    return detections

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="GPR-X Multiclass Detector", layout="wide")
st.title("📡 GPR-X Intelligent Multi-Target Detector")
st.write("Upload a radargram to identify Cavities, Bricks, and Metal Pipes.")

uploaded_file = st.sidebar.file_uploader("Upload PNG/JPG Radargram", type=["png", "jpg", "jpeg"])
conf_threshold = st.sidebar.slider("Detection Confidence", 0.70, 0.99, 0.92)

if uploaded_file and model:
    input_img = Image.open(uploaded_file).convert('RGB')
    
    if st.sidebar.button("🔍 Run Full Scan"):
        start_time = time.time()
        results = run_detection(input_img, conf_threshold)
        duration = time.time() - start_time
        
        # Plotting results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(input_img)
        
        # Styling mapping
        # 1: Cavity (Blue), 2: Brick (White), 3: Metal Pipe (Cyan)
        styles = {
            1: ("Cavity", "blue"),
            2: ("Brick", "white"),
            3: ("Metal Pipe", "cyan")
        }
        
        for det in results:
            label, color = styles[det['class']]
            x, y, w, h = det['box']
            
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, f"{label} {det['conf']:.2f}", color=color, fontweight='bold', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        
        plt.axis('off')
        st.pyplot(fig)
        st.success(f"Scan complete in {duration:.1f}s. Identified {len(results)} targets.")

elif not model:
    st.error("Missing AI Models! Please upload 'svm_model.pkl' and 'scaler.pkl' to the app directory.")
