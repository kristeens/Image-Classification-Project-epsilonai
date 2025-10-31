
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.warning(str(e))

st.set_page_config(page_title="🍔 Fast Food Classifier", layout="centered")

st.title("🍕 Fast Food Image Classifier")
st.write("Upload a food image and the model will predict its class!")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            r"final_fastfood_DenseNet.h5"
        )
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()


class_names = [
    "Burger", "Pizza", "Hotdog", "Fries", "Sandwich",
    "Taco", "Nuggets", "Donut", "Pasta", "Salad"
]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # same rescaling as training

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display result
    st.markdown(f"### 🍟 Prediction: **{predicted_class}**")
    
