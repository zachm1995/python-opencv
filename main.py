# Imports
import cv2
import numpy as np
import pytesseract as pt
import threading
from time import sleep

# Functions

def initialize_webcam_feed():
    cv2.namedWindow("Webcam Feed")
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    return camera

def update_output_window(camera):
    while camera.isOpened():
        resp, frame = camera.read()
        if resp:
            cv2.imshow("Webcam Feed", frame)
            
        # Wait for user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Program

# Initialize webcam
# camera = initialize_webcam_feed()

# Open static image
image = cv2.imread('image.jpg')
h, w, _ = image.shape
image = cv2.resize(image, (int(w * .2), int(h * .2)))

# Create a thread to update the output window with frames from the camera
# cameraOutputThread = threading.Thread(update_output_window(camera), name="cameraOutput")

cv2.imshow("Image", image)

kernel = np.ones((5, 5), np.uint8)

imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 100)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

cv2.imshow("Grey", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Dialated", imgDialation)
cv2.imshow("Eroded", imgEroded)

cv2.waitKey(0)
cv2.destroyAllWindows()