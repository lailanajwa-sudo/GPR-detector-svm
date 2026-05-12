import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. ASSET LOADING ---
def load_assets():
    try:
        # Loading the SVM model and scaler trained in Colab
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Model files missing: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING (Same as MATLAB/Colab Training) ---
def preprocess_roi(roi_img):
    # Grayscale -> Resize to 120x100 (exactly 12,000 features)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray equivalent
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Histogram Equalization to boost hyperbolic contrast (histeq)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    
    return img_eq.flatten().reshape(1, -1)

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="GPR Interactive Classifier", layout="wide")
st.title("📡 Interactive GPR Target Classifier")
st.write("Manually move the box over a hyperbolic signature to classify it.")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # --- INTERACTIVE SLIDERS ---
    st.sidebar.header("🕹️ Position Control")
    pos_x = st.sidebar.slider("Move Box (X Direction)", 0, w - 120, int(w/2))
    pos_y = st.sidebar.slider("Move Box (Y Direction)", 0, h - 100, int(h/2))
    
    # Define Target Size (Matches Training)
    box_w, box_h = 120, 100
    
    # 1. Extract the ROI based on user sliders
    roi = img.crop((pos_x, pos_y, pos_x + box_w, pos_y + box_h))
    
    # 2. Preprocess and Predict
    features = preprocess_roi(roi)
    features_scaled = scaler.transform(features)
    
    probs = model.predict_proba(features_scaled)[0]
    classes = ["Cavity", "Brick", "Metal Pipe"]
    best_idx = np.argmax(probs)
    prediction = classes[best_idx]
    confidence = probs[best_idx]
    
    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        # Color mapping: Cavity(Blue), Brick(White), Metal(Cyan)
        styles = {"Cavity": "blue", "Brick": "white", "Metal Pipe": "cyan"}
        box_color = styles[prediction]
        
        # Draw the manual bounding box
        rect = patches.Rectangle((pos_x, pos_y), box_w, box_h, linewidth=3, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Targeting: {pos_x}, {pos_y}")
        plt.axis('off')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Classification Result")
        st.metric(label="Detected Object", value=prediction)
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        
        st.write("**Probability Breakdown:**")
        for i, prob in enumerate(probs):
            st.write(f"{classes[i]}: {prob*100:.1f}%")
            st.progress(float(prob))

        st.info("💡 Tip: Align the box so the hyperbola curve is centered inside.")

else:
    st.info("Please upload a radargram to begin interactive classification.")
