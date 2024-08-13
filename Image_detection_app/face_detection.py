import streamlit as st
from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image

# Initialize the MTCNN detector
detector = MTCNN()

# Streamlit app title
st.title("Face Detection App")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect faces in the image
    output = detector.detect_faces(img)

    # Draw rectangles and circles on the detected faces
    for i in output:
        x, y, width, height = i['box']

        left_eyeX, left_eyeY = i['keypoints']['left_eye']
        right_eyeX, right_eyeY = i['keypoints']['right_eye']
        noseX, noseY = i['keypoints']['nose']
        mouth_leftX, mouth_leftY = i['keypoints']['mouth_left']
        mouth_rightX, mouth_rightY = i['keypoints']['mouth_right']

        # Draw circles on keypoints
        cv2.circle(img, center=(left_eyeX, left_eyeY), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(right_eyeX, right_eyeY), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(noseX, noseY), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(mouth_leftX, mouth_leftY), color=(255, 0, 0), thickness=3, radius=2)
        cv2.circle(img, center=(mouth_rightX, mouth_rightY), color=(255, 0, 0), thickness=3, radius=2)

        # Draw rectangle around the face
        cv2.rectangle(img, pt1=(x, y), pt2=(x+width, y+height), color=(255, 0, 0), thickness=3)

    # Convert BGR to RGB format to display with Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with detections
    st.image(img_rgb, caption="Processed Image", use_column_width=True)
