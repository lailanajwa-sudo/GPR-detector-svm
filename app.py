import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Setup the page title
st.set_page_config(page_title="GPR Detector")
st.title("GPR Radargram Object Detection")
st.write("Upload a Radargram image to find Cavities, Metal Pipes, or Bricks.")

# 2. Load your trained 'best.pt' model
@st.cache_resource # This keeps the model in memory so it stays fast
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. Create the File Uploader
uploaded_file = st.file_uploader("Upload Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    
    # Show the user what they uploaded
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Detecting objects...")

    # 4. Run the YOLOv8 model on the image
    img_array = np.array(image)
    results = model.predict(source=img_array, conf=0.25)
    
    # 5. Draw the boxes on the image and show it
    res_plotted = results[0].plot() 
    st.image(res_plotted, caption='Detection Results', use_container_width=True)
    
    # 6. Print the results in text format
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        st.success(f"Found: {label} (Confidence: {conf:.2f})")