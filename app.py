import streamlit as st
import cv2
import numpy as np




# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered"
)

# Title
st.title("👤 Face Detection using OpenCV")

st.write(
    "Upload an image and this application will detect "
    "human faces using OpenCV Haar Cascade."
)

# Load Haar Cascade
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Copy image
    output_image = image.copy()

    # Draw rectangle around each face
    for (x, y, w, h) in faces:

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Convert BGR to RGB
    output_rgb = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )

    # Display result
    st.image(
        output_rgb,
        caption=f"Detected Faces: {len(faces)}",
        use_container_width=True
    )

    # Display message
    if len(faces) > 0:

        st.success(
            f"✅ {len(faces)} face(s) detected!"
        )

    else:

        st.warning(
            "⚠️ No faces detected."
        )