import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model(r'Project_Model.h5')

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess the image
def preprocess_image(image, img_size=128):
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to classify the image
def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction, axis=-1)[0]

# Streamlit app
st.title("Face Mask Detection")

# Create tabs
tab1, tab2 = st.tabs(["Image Upload", "Real-time Detection"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))  # Ensure image is in RGB format
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            if face.size != 0:  # Ensure face is not empty
                label = classify_image(face)
                text = "With Mask" if label == 0 else "Without Mask"
                color = (0, 255, 0) if label == 0 else (255, 0, 0)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        st.image(image, caption='Processed Image', use_column_width=True)

with tab2:
    st.header("Real-time Detection")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame_rgb[y:y+h, x:x+w]
            if face.size != 0:  # Ensure face is not empty
                label = classify_image(face)
                text = "With Mask" if label == 0 else "Without Mask"
                color = (0, 255, 0) if label == 0 else (255, 0, 0)
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_rgb, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME_WINDOW.image(frame_rgb)

    cap.release()