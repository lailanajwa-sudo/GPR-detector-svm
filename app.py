import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. ADVANCED PREPROCESSING ---
def preprocess_roi(roi_img):
    # A. Convert to Grayscale
    roi = ImageOps.grayscale(roi_img)
    
    # B. ENHANCEMENT: Sharpen the hyperbola to make the 'legs' clearer
    # This helps the SVM distinguish between Brick and Metal
    roi = roi.filter(ImageFilter.SHARPEN)
    
    # C. Resize to match 12,000 features
    roi_resized = roi.resize((120, 100))
    
    # D. Robust Normalization (Min-Max Scaling)
    img_np = np.array(roi_resized).astype(np.float64)
    img_min = np.percentile(img_np, 2) # Use 2nd percentile to ignore dark noise
    img_max = np.percentile(img_np, 98) # Use 98th percentile to ignore bright spots
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
    img_norm = np.clip(img_norm, 0, 1) # Ensure values are exactly 0 to 1
    
    # E. Transform using training scaler
    features = img_norm.flatten().reshape(1, -1)
    return scaler.transform(features)

# --- 3. UI ---
st.set_page_config(layout="centered")
st.title("📡 GPR Advanced Classifier")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    BW, BH = 80, 60 # Fixed box size for consistency

    st.write("### Position the Box")
    pos_x = st.slider("X Position", 0, w - BW, int(w/2))
    pos_y = st.slider("Y Position", 0, h - BH, int(h/2))

    # Small Preview
    _, mid_col, _ = st.columns([1, 4, 1])
    with mid_col:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(img)
        rect = patches.Rectangle((pos_x, pos_y), BW, BH, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    if st.button("🚀 ANALYZE TARGET", use_container_width=True):
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Calculate signal strength
        roi_np = np.array(ImageOps.grayscale(roi))
        signal_strength = np.std(roi_np)

        if signal_strength < 10:
            st.warning(f"Low signal ({signal_strength:.1f}). This area is likely Cavity or Background.")
        
        with st.spinner('Extracting Patterns...'):
            features_scaled = preprocess_roi(roi)
            probs = model.predict_proba(features_scaled)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            idx = np.argmax(probs)
            
            st.divider()
            st.success(f"### Result: {classes[idx]}")
            st.metric("Confidence", f"{probs[idx]*100:.1f}%")
            
            # Show the actual ROI being analyzed to check clarity
            st.write("Processed ROI (What the AI sees):")
            st.image(roi.resize((240, 200)), width=150)
