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
    # Grayscale -> Sharpen -> Resize
    roi = ImageOps.grayscale(roi_img).filter(ImageFilter.SHARPEN)
    roi_resized = roi.resize((120, 100))
    
    img_np = np.array(roi_resized).astype(np.float64)
    # Robust scaling (mat2gray)
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    features = img_norm.flatten().reshape(1, -1)
    return scaler.transform(features)

# --- 3. UI SETUP ---
st.set_page_config(layout="centered")
st.title("📡 GPR Target Classifier")

if not model:
    st.error("⚠️ Files 'svm_model.pkl' or 'scaler.pkl' not found in directory!")

uploaded_file = st.file_uploader("Step 1: Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # --- BUTTON AT THE TOP (Always Visible) ---
    st.divider()
    analyze_now = st.button("🚀 START CLASSIFICATION", use_container_width=True, type="primary")
    st.divider()

    # --- CONTROLS ---
    BW, BH = 80, 60 # Fixed box size
    
    st.write("### Step 2: Position the Box")
    # Edge protection: ensure sliders don't go out of bounds
    pos_x = st.slider("X (Horizontal)", 0, max(1, w - BW), int(w/2))
    pos_y = st.slider("Y (Vertical)", 0, max(1, h - BH), int(h/2))

    # --- SMALL PREVIEW ---
    # Centering the image preview
    col_a, col_img, col_b = st.columns([1, 6, 1])
    with col_img:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(img)
        rect = patches.Rectangle((pos_x, pos_y), BW, BH, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 4. CLASSIFICATION LOGIC ---
    if analyze_now:
        # Define crop area
        left = pos_x
        top = pos_y
        right = pos_x + BW
        bottom = pos_y + BH
        
        roi = img.crop((left, top, right, bottom))
        
        # Check if the selection is valid
        roi_np = np.array(ImageOps.grayscale(roi))
        if np.std(roi_np) < 5:
            st.warning("⚠️ Area is too blank. Move the red box onto a hyperbolic pattern.")
        else:
            with st.spinner('AI is analyzing the signal...'):
                processed = preprocess_roi(roi)
                probs = model.predict_proba(processed)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                result_idx = np.argmax(probs)
                
                # RESULTS DISPLAY
                st.markdown(f"## Target: {classes[result_idx]}")
                st.progress(float(probs[result_idx]))
                
                # Show probabilities
                c1, c2, c3 = st.columns(3)
                c1.metric("Cavity", f"{probs[0]*100:.1f}%")
                c2.metric("Brick", f"{probs[1]*100:.1f}%")
                c3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
                
                # Show the AI's view
                st.write("AI Zoom View:")
                st.image(roi, width=150)
