import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- ASSET LOADING ---
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- PREPROCESSING (Same as Training) ---
def preprocess_roi(roi_img):
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    # Normalize
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    # Equalize to highlight hyperbolic edges
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    return img_eq.flatten().reshape(1, -1)

# --- UI SETTINGS ---
st.title("📡 GPR Hyperbolic Signature Detector")
uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

# Sidebar Controls for "Clean" Detection
st.sidebar.header("Hyperbola Tuning")
conf_threshold = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.94)
# Surface Offset helps avoid the top horizontal direct coupling line
surface_offset = st.sidebar.slider("Vertical Offset (Skip Surface)", 0, 200, 80)

if uploaded_file and model:
    input_img = Image.open(uploaded_file).convert('RGB')
    st.image(input_img, caption="Original GPR Data")
    
    if st.button("🔍 Detect Hyperbolas"):
        w, h = input_img.size
        roi_w, roi_h = 120, 100
        stride = 30 
        detections = []
        
        progress = st.progress(0)
        
        # We start 'y' at surface_offset to avoid the direct coupling line at the top
        y_range = range(surface_offset, h - roi_h, stride)
        x_range = range(0, w - roi_w, stride)
        
        total_steps = len(y_range) * len(x_range)
        step = 0

        for y in y_range:
            for x in x_range:
                step += 1
                progress.progress(step / total_steps)
                
                roi = input_img.crop((x, y, x + roi_w, y + roi_h))
                
                # SIG-CHECK: Only scan if the area isn't "flat" (prevents background boxes)
                if np.std(np.array(roi)) > 5: 
                    feat = preprocess_roi(roi)
                    feat_scaled = scaler.transform(feat)
                    
                    probs = model.predict_proba(feat_scaled)[0]
                    idx = np.argmax(probs)
                    conf = probs[idx]
                    
                    if conf >= conf_threshold:
                        detections.append({'box': [x, y], 'class': idx + 1, 'conf': conf})

        # --- DRAWING RESULTS ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(input_img)
        
        # Match colors to your provided screenshots
        styles = {
            1: ("Cavity", "blue"),      # Blue box for cavity
            2: ("Brick", "white"),      # White box for brick
            3: ("Metal Pipe", "cyan")   # Cyan/Light Blue for metal
        }
        
        for d in detections:
            label, color = styles[d['class']]
            bx, by = d['box']
            
            # Draw bounding box only around the signature
            rect = patches.Rectangle((bx, by), roi_w, roi_h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(bx, by - 5, f"{label} {d['conf']:.2f}", color=color, fontsize=7, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
        plt.axis('off')
        st.pyplot(fig)
        st.success(f"Detection complete. Found {len(detections)} hyperbolic signatures.")
