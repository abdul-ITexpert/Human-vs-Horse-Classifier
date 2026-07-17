import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Page Configuration & UI Title
st.set_page_config(page_title="Horse vs Human Classifier", page_icon="🐎", layout="centered")
st.title("🐎 Horse vs Human Classifier 🧍")
st.write("Upload an image, and our Transfer Learning model will predict what it is!")

# 2. Cache the Model Loading function so it stays persistent in RAM
@st.cache_resource
def load_my_model():
    # Loads the saved transfer learning model file
    return tf.keras.models.load_model("horse_human_mobilenet.keras")

with st.spinner("Loading AI Model into memory... Please wait."):
    model = load_my_model()

# 3. File Upload Widget
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the user's image cleanly on screen
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("---")
    st.write("🤖 **AI Analysis running...**")
    
    # 4. Match the exact Training Preprocessing Pipeline
    # Convert image to RGB format (handles transparent PNG conversions)
    img_rgb = image.convert("RGB")
    # Resize image to match input shape (64x64)
    img_resized = img_rgb.resize((64, 64))
    # Convert image array to numpy matrix
    img_array = np.array(img_resized, dtype=np.float32)
    # Add batch dimension -> shape goes from (64,64,3) to (1,64,64,3)
    img_tensor = np.expand_dims(img_array, axis=0)
    # Scale pixels between -1 and 1 matching MobileNetV2
    processed_tensor = preprocess_input(img_tensor)
    
    # 5. Model Prediction Inference
    prediction = model.predict(processed_tensor)
    raw_score = prediction[0][0]  # Extracts the scalar sigmoid probability
    
    # 6. Display Clean Interactive Results
    if raw_score > 0.5:
        confidence = raw_score * 100
        st.success(f"### Prediction: **Human** 🧍")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
    else:
        confidence = (1 - raw_score) * 100
        st.success(f"### Prediction: **Horse** 🐎")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
