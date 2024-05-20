import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from deepface import DeepFace

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = ["stressless", "stressful"]

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Set the title and subtitle
st.title("DHV AI Stress Analysis")
st.subheader("การประเมินความเครียดจากสีหน้าด้วยภาพถ่าย Selfie --สำหรับผู้ใหญ่")
st.write("โปรดอัปโหลดภาพถ่ายหน้าตรง ไม่ปิดบังส่วนใดของใบหน้า")

# Define a function to convert the uploaded file to a numpy array
def load_image(image_file):
    # Load the image using PIL
    image = Image.open(image_file)
    # Convert the image to RGB mode if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to 224x224
    image = image.resize((224, 224), resample=Image.LANCZOS)
    # Convert the image to a numpy array
    image_array = np.array(image)
    return image_array

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image_array = load_image(image_file)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image and prediction
    caption = f"<h1 style='font-size: 24px; color: {'blue' if class_name == 'stressless' else 'red'}'>Class: {class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}</h1>"
    st.image(image_array, caption=None)
    st.markdown(caption, unsafe_allow_html=True)
    if class_name == "stressless":
        st.markdown("<h1 style='font-size: 24px;'>You've done good so far.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 24px;'>Take a deep breath! Sit back and relax. Try exercise, vacation, entertainment, or consult your physician.</h1>", unsafe_allow_html=True)

    # Perform stress detection using DeepFace
    results = DeepFace.analyze(image_array, actions=['age'])
    # Iterate over the list and find the dictionary with 'age' information
    predicted_age = None

    if results:
        # Assume the first element in the list has the 'age' information
        first_result = results[0]

        # Check if the first_result is a dictionary and contains the 'age' key
        if isinstance(first_result, dict) and 'age' in first_result:
            predicted_age = first_result['age']

    st.markdown(f"<h1>Predicted Age: {predicted_age}</h1>", unsafe_allow_html=True)

# Create a file uploader and predict when a file is uploaded
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","webp"])
if uploaded_file is not None:
    predict_image(uploaded_file)