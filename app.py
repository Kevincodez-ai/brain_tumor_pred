import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("enhanced braintumor detection.h5")
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🧠 Enhanced Brain Tumor Detection")
st.markdown("Upload an MRI scan to detect if there's a **brain tumor**.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)[0][0]
        
        if prediction > 0.5:
            st.error(f"🧠 Tumor Detected with {prediction*100:.2f}% confidence.")
        else:
            st.success(f"✅ No Tumor Detected with {(1 - prediction)*100:.2f}% confidence.")
