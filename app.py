import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error("Error: 'svm_model.pkl' or 'scaler.pkl' not found. Please upload the files from Colab.")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING LOGIC ---
def preprocess_roi(roi_img, gain):
    # 1. Enhance Contrast (Mimics BEMD enhancement)
    # Higher gain helps pull Brick/Cavity signals out of the noise
    enhancer = ImageEnhance.Contrast(roi_img)
    roi_img = enhancer.enhance(gain)
    
    # 2. Convert to Grayscale and Resize to match training (120x100)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # 3. mat2gray Normalization (0.0 to 1.0)
    img_min = np.min(img_np)
    img_max = np.max(img_np)
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    
    # 4. Flatten and Apply Training Scaler
    flat_data = img_norm.flatten().reshape(1, -1)
    scaled_data = scaler.transform(flat_data)
    return scaled_data

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="GPR Target Classifier", layout="centered")

st.title("📡 GPR Hyperbolic Pattern Classifier")
st.write("Target specific anomalies by moving the red bounding box.")

uploaded_file = st.file_uploader("Step 1: Upload Radargram Image", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Load Image
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # Configuration
    BW, BH = 80, 60 # Fixed box size for consistent feature extraction

    st.divider()
    
    # --- CONTROLS ---
    st.subheader("Step 2: Position & Sensitivity")
    col_pos, col_sens = st.columns([2, 1])
    
    with col_pos:
        pos_x = st.slider("Horizontal (X)", 0, max(1, w - BW), int(w/2))
        pos_y = st.slider("Vertical (Y)", 0, max(1, h - BH), int(h/2))
    
    with col_sens:
        # Gain of 1.5-2.0 is usually best for Brick/Cavity
        gain = st.slider("Signal Gain", 0.5, 4.0, 1.5, help="Increase if pattern is faint (Brick/Cavity)")

    # --- SMALL PREVIEW ---
    # Centering the preview using columns
    _, mid_col, _ = st.columns([1, 4, 1])
    with mid_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(img)
        # Red box representing the 80x60 crop area
        rect = patches.Rectangle((pos_x, pos_y), BW, BH, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- CLASSIFICATION ---
    if st.button("🚀 ANALYZE TARGET", use_container_width=True, type="primary"):
        # Crop the ROI
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        with st.spinner("Processing Signal..."):
            # Preprocess and Predict
            processed_features = preprocess_roi(roi, gain)
            probs = model.predict_proba(processed_features)[0]
            
            # Map labels (match your Colab: 1=Cavity, 2=Brick, 3=Metal)
            classes = ["Cavity", "Brick", "Metal Pipe"]
            best_idx = np.argmax(probs)
            
            st.divider()
            
            # Show Results
            st.subheader(f"Detection: {classes[best_idx]}")
            
            # Detail view
            c1, c2, c3 = st.columns(3)
            c1.metric("Cavity", f"{probs[0]*100:.1f}%")
            c2.metric("Brick", f"{probs[1]*100:.1f}%")
            c3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
            
            # Show the "AI View"
            st.write("**AI Zoom View (Processed):**")
            st.image(roi.resize((240, 180)), width=150)
            
            if probs[best_idx] < 0.5:
                st.warning("Low confidence score. Try adjusting the Signal Gain or centering the box exactly on the hyperbola peak.")

elif not model:
    st.info("Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the app folder.")
