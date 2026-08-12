import cv2
import os
import urllib.request
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)

# --------------------------------------------------
# Load Haar Cascade
# --------------------------------------------------

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

print(
    "Face detector loaded:",
    not face_detector.empty()
)

# --------------------------------------------------
# Image Acquisition
# --------------------------------------------------

IMAGE_PATH = "input.jpeg"

FALLBACK_FACE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/data/lena.jpg"
)

FALLBACK_FACE_PATH = "sample_face.jpg"

if os.path.exists(IMAGE_PATH):

    face_image_path = IMAGE_PATH

    print(
        f"Using your image: {face_image_path}"
    )

else:

    print("input.jpeg not found.")
    print("Downloading sample face image...")

    if not os.path.exists(FALLBACK_FACE_PATH):

        urllib.request.urlretrieve(
            FALLBACK_FACE_URL,
            FALLBACK_FACE_PATH
        )

    face_image_path = FALLBACK_FACE_PATH

    print(
        f"Using sample image: {face_image_path}"
    )

# --------------------------------------------------
# Read Image
# --------------------------------------------------

face_image = cv2.imread(face_image_path)

if face_image is None:

    print("Error: Could not load the image.")
    exit()

print(
    "Image loaded:",
    face_image_path,
    "| Shape:",
    face_image.shape
)

# --------------------------------------------------
# Convert Image to Grayscale
# --------------------------------------------------

gray_image = cv2.cvtColor(
    face_image,
    cv2.COLOR_BGR2GRAY
)

# --------------------------------------------------
# Detect Faces
# --------------------------------------------------

faces = face_detector.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(
    "Faces detected:",
    len(faces)
)

# --------------------------------------------------
# Draw Rectangle Around Faces
# --------------------------------------------------

output_image = face_image.copy()

for (x, y, w, h) in faces:

    cv2.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

# --------------------------------------------------
# Save Output Image
# --------------------------------------------------

cv2.imwrite(
    "output.jpg",
    output_image
)

print(
    "Output saved as: output.jpg"
)

# --------------------------------------------------
# Display Image
# --------------------------------------------------

output_rgb = cv2.cvtColor(
    output_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 7))

plt.imshow(output_rgb)

plt.axis("off")

plt.title(
    f"Detected Faces: {len(faces)}"
)

plt.show()