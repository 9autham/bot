import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../potatoes.h5")
MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

st.title("POTATO LEAF DISEASE PREDICTION")

uploaded_image = st.file_uploader("Upload Potato Leaf Image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    bytes_data = uploaded_image.getvalue()
    # Resize the image to (256, 256)
    image = np.array(Image.open(BytesIO(bytes_data)).resize((256, 256)))
    img_batch = np.expand_dims(image, 0)
    try:
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        up_image = Image.open(BytesIO(bytes_data))
        st.image(up_image, width=300, use_column_width=True)
        st.success("Leaf is in {} condition and Prediction Confidence = {}%".format(predicted_class,
                                                                                     round(confidence * 100, 2)))
    except Exception as e:
        st.title("Invalid Image")
        print(e)

st.markdown("<h3 style='text-align: center;'>Hopes</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>Author</p>", unsafe_allow_html=True)
