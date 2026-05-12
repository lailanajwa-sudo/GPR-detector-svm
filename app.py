import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Ensure these filenames match exactly what you downloaded from Colab
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

def preprocess_roi(roi_img):
    # 1. Convert to Grayscale
    roi_gray = ImageOps.grayscale(roi_img)
    
    # 2. Resize to 120x100 (MUST match the 12,000 features in your Excel)
    roi_resized = roi_gray.resize((120, 100))
    
    # 3. Convert to Numpy and Normalize (mat2gray equivalent)
    img_np = np.array(roi_resized).astype(np.float64)
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # 4. Flatten to a row vector
    flat_data = img_norm.flatten().reshape(1, -1)
    
    # 5. USE THE SCALER (This is the most important step!)
    # This transforms the live image to match the 'StandardScaler' from Colab
    scaled_data = scaler.transform(flat_data)
    
    return scaled_data

# --- INSIDE YOUR ANALYZE BUTTON ---
if st.button("🚀 Analyze"):
    roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
    
    # Check if the box is just empty background
    if np.std(np.array(roi)) < 10:
        st.warning("This looks like empty background/soil. No signal detected.")
    else:
        # Preprocess using the function above
        processed_features = preprocess_roi(roi)
        
        # Predict
        probs = model.predict_proba(processed_features)[0]
        classes = ["Cavity", "Brick", "Metal Pipe"]
        prediction = classes[np.argmax(probs)]
        
        st.success(f"Result: {prediction}")
        # Show percentages so you can see if it was close
        for i, class_name in enumerate(classes):
            st.write(f"{class_name}: {probs[i]*100:.1f}%")
