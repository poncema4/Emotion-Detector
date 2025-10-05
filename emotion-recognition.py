import cv2
import numpy as np
import tensorflow as tf
import streamlit as st

st.title("Emotion Recognition App")
st.write("Turn on your webcam to detect emotions.")

emotion_model = tf.keras.models.load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

uploaded_file = st.camera_input("Take a picture")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.getbuffer()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
                .detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        roi = cv2.resize(gray[y:y+h, x:x+w], (64,64))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        pred = emotion_model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(pred)]
        st.write(f"**Detected Emotion:** {emotion}")