import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('svm_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 2. UI CONFIG ---
st.set_page_config(layout="centered")
st.title("📡 GPR Pattern Classifier")

# --- 3. THE ANALYZE BUTTON (Moved to the Top for Visibility) ---
# We use a placeholder to ensure the button is always at the top
st.info("Upload a file, position the red box, then click Analyze.")
analyze_btn = st.button("🚀 CLICK HERE TO ANALYZE", use_container_width=True, type="primary")

# --- 4. UPLOAD & SETUP ---
uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # Target Box Size (Fixed)
    BW, BH = 80, 60

    # SLIDERS
    st.write("### Target Positioning")
    # Safety: ensure sliders don't allow box to go outside image
    x = st.slider("X Position", 0, max(1, w - BW), int(w/2))
    y = st.slider("Y Position", 0, max(1, h - BH), int(h/2))
    gain = st.slider("Signal Gain (Increase for Brick/Cavity)", 0.5, 4.0, 1.5)

    # --- SMALL PREVIEW ---
    # Centering the preview so it's not too big
    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_m:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(img)
        rect = patches.Rectangle((x, y), BW, BH, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # --- 5. CLASSIFICATION LOGIC ---
    if analyze_btn:
        with st.spinner('Analyzing...'):
            # Step 1: Crop and Enhance
            roi = img.crop((x, y, x + BW, y + BH))
            enhancer = ImageEnhance.Contrast(roi)
            roi_enhanced = enhancer.enhance(gain)
            
            # Step 2: Convert to Grayscale & Resize to 120x100
            roi_gray = ImageOps.grayscale(roi_enhanced).resize((120, 100))
            img_np = np.array(roi_gray).astype(np.float64)
            
            # Step 3: Normalize (mat2gray)
            img_min, img_max = np.min(img_np), np.max(img_np)
            img_norm = (img_np - img_min) / (img_max - img_min + 1e-7)
            
            # Step 4: Scale and Predict
            features = scaler.transform(img_norm.flatten().reshape(1, -1))
            probs = model.predict_proba(features)[0]
            
            # Labels (Match your Colab: 1=Cavity, 2=Brick, 3=Metal Pipe)
            classes = ["Cavity", "Brick", "Metal Pipe"]
            idx = np.argmax(probs)

            # RESULTS
            st.divider()
            st.success(f"### RESULT: {classes[idx]}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Cavity", f"{probs[0]*100:.1f}%")
            c2.metric("Brick", f"{probs[1]*100:.1f}%")
            c3.metric("Metal Pipe", f"{probs[2]*100:.1f}%")
            
            st.write("**AI Zoom View:**")
            st.image(roi.resize((240, 180)), width=200)

elif not model:
    st.error("⚠️ Files missing! Please upload 'svm_model.pkl' and 'scaler.pkl' to the same folder.")
