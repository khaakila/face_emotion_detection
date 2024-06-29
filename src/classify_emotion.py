import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model for emotion classification
model = load_model('../models/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def classify_emotion(face_image):
    # Prepare the face image for classification
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    face_image = face_image / 255.0
    prediction = model.predict(face_image)
    return emotion_labels[np.argmax(prediction)]
