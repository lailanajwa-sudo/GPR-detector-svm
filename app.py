import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Loading the SVM model and scaler trained in Colab
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING (Standardizing for 12,000 features) ---
def preprocess_roi(roi_img):
    # Grayscale -> Resize to 120x100 to match training data
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray: Scale pixels between 0 and 1
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Histogram Equalization: Boosts the hyperbola curve visibility
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    
    return img_eq.flatten().reshape(1, -1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR Manual Classifier", layout="wide")
st.title("📡 GPR Hyperbola Manual Classifier")
st.write("Use the sliders below to move the box over a hyperbolic signature.")

# Check if model files are present
if model is None:
    st.error("⚠️ Error: 'svm_model.pkl' or 'scaler.pkl' not found in the directory!")
    st.stop()

uploaded_file = st.file_uploader("Step 1: Upload your Radargram (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    st.divider()
    
    # --- STEP 2: POSITION SLIDERS (IN THE MAIN BODY) ---
    st.subheader("Step 2: Align the Target Box")
    col_x, col_y = st.columns(2)
    with col_x:
        pos_x = st.slider("Horizontal Position (X)", 0, w - 120, int(w/2))
    with col_y:
        pos_y = st.slider("Vertical Position (Y)", 0, h - 100, int(h/2))
    
    # Define detection size (Matches your BEMD parameters)
    box_w, box_h = 120, 100
    
    # 1. Crop the ROI
    roi = img.crop((pos_x, pos_y, pos_x + box_w, pos_y + box_h))
    
    # 2. Predict using the SVM
    features = preprocess_roi(roi)
    features_scaled = scaler.transform(features)
    
    probs = model.predict_proba(features_scaled)[0]
    classes = ["Cavity", "Brick", "Metal Pipe"]
    best_idx = np.argmax(probs)
    prediction = classes[best_idx]
    confidence = probs[best_idx]
    
    # --- STEP 3: DISPLAY RESULTS ---
    st.divider()
    res_col, img_col = st.columns([1, 2])
    
    with res_col:
        st.subheader("Classification Result")
        # Color mapping: Cavity(Blue), Brick(White), Metal(Cyan)
        styles = {"Cavity": "blue", "Brick": "white", "Metal Pipe": "cyan"}
        current_color = styles[prediction]
        
        st.info(f"The object inside the box is likely a: **{prediction}**")
        st.metric("Confidence", f"{confidence*100:.1f}%")
        
        st.write("**Full Analysis:**")
        for i, prob in enumerate(probs):
            st.write(f"{classes[i]}: {prob*100:.1f}%")
            st.progress(float(prob))

    with img_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(img)
        
        # Draw the target box
        rect = patches.Rectangle((pos_x, pos_y), box_w, box_h, 
                                linewidth=4, edgecolor=current_color, facecolor='none')
        ax.add_patch(rect)
        
        # Label above the box
        ax.text(pos_x, pos_y - 10, f"{prediction} ({confidence*100:.0f}%)", 
                color=current_color, fontweight='bold', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        
        plt.axis('off')
        st.pyplot(fig)

else:
    st.info("Please upload a radargram to start.")
