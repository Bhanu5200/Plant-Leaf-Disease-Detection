import streamlit as st

import tensorflow as tf

import numpy as np

from PIL import Image

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="")

st.title("Plant Leaf Disease Detection")

st.write("Upload a leaf image to detect disease")

# Load model 

@st.cache_resource

def load_model():

    model = tf.keras.models.load_model('model.h5')

    return model

model = load_model()

# Upload image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess

    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Predict

    prediction = model.predict(img_array)

    class_names = ['Healthy', 'Rust', 'Spot'] # Change as per your classes

    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}") 
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")