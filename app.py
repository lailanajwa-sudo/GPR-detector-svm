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
        # These MUST be the new files you downloaded from Colab
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Grayscale and Resize to match your 12,000 feature training (120x100)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray normalization
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Flatten to row vector
    flat_data = img_norm.flatten().reshape(1, -1)
    
    # IMPORTANT: Use the scaler from your training
    scaled_data = scaler.transform(flat_data)
    return scaled_data

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR Target Finder", layout="wide")
st.title("📡 GPR Hyperbola Manual Classifier")
st.write("Align the yellow box with a hyperbolic peak, then click Analyze.")

uploaded_file = st.file_uploader("Step 1: Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Load and get dimensions
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # --- STEP 2: POSITIONING SLIDERS ---
    # These update the box position on the preview image
    st.subheader("Step 2: Position the Bounding Box")
    col_x, col_y = st.columns(2)
    with col_x:
        pos_x = st.slider("X Position (Horizontal)", 0, w - 120, int(w/2))
    with col_y:
        pos_y = st.slider("Y Position (Vertical)", 0, h - 100, int(h/2))
    
    # --- PREVIEW PLOT ---
    # This shows you WHERE the box is in real-time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    # The box size is fixed to 120x100 to match your training features
    rect = patches.Rectangle((pos_x, pos_y), 120, 100, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # --- STEP 3: THE ANALYZE BUTTON ---
    # Prediction only happens when you click this
    if st.button("🚀 Analyze Selected Target"):
        roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
        roi_np = np.array(roi)
        
        # FILTER: If standard deviation is low, it's just background/noise
        if np.std(roi_np) < 10:
            st.warning("⚠️ Warning: The box is currently over a blank/flat area. No strong signal detected.")
        else:
            with st.spinner("Processing BEMD Features..."):
                # Preprocess and Predict
                processed_features = preprocess_roi(roi)
                probs = model.predict_proba(processed_features)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                best_idx = np.argmax(probs)
                
                st.divider()
                # Display final result
                st.subheader(f"Detection Result: {classes[best_idx]}")
                st.metric("Confidence Score", f"{probs[best_idx]*100:.1f}%")
                
                # Confidence Breakdown
                for i, p in enumerate(probs):
                    st.write(f"{classes[i]}: {p*100:.1f}%")
                    st.progress(float(p))
