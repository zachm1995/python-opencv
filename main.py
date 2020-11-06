# Imports
import cv2
import numpy as np
import pytesseract as pt
import threading

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

# From Murtaza
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty():
    pass

def initialize_trackbar(settings):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    for setting in settings:
        cv2.createTrackbar(f"{setting['name']} Min", "Trackbars", setting['min'], setting['max'], empty)
        cv2.createTrackbar(f"{setting['name']} Max", "Trackbars", setting['max'], setting['max'], empty)

# Program

# Initialize webcam
# camera = initialize_webcam_feed()

# Open static image
image = cv2.imread('image.jpg')

# Get image size
h, w, _ = image.shape

# Reduce size by 20%
imageSmall = cv2.resize(image, (int(w * .2), int(h * .2)))

# Create a thread to update the output window with frames from the camera
# cameraOutputThread = threading.Thread(update_output_window(camera), name="cameraOutput")

# cv2.imshow("Image", image)

# Create 5x5 array of 1s
# kernel = np.ones((5, 5), np.uint8)

# Manipulate image
# imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
# imgCanny = cv2.Canny(imgBlur, 100, 100)
# imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# imgHorizontalStack = np.hstack((image, image))

# cv2.imshow("Horizontal", imgHorizontalStack)

# cv2.imshow("Grey", imgGray)
# cv2.imshow("Blur", imgBlur)
# cv2.imshow("Canny", imgCanny)
# cv2.imshow("Dialated", imgDialation)
# cv2.imshow("Eroded", imgEroded)

# Stack images
# imgStack = stackImages(.5, [image, imageSmall, image, imageSmall])
# cv2.imshow("Stack", imgStack)

# Color detection
settings = [{
    "name": "Hue",
    "min": 0,
    "max" : 179
}, {
    "name": "Sat",
    "min": 0,
    "max" : 255
}, {
    "name": "Val",
    "min": 0,
    "max" : 255
}]

initialize_trackbar(settings)

while True:
    imgHSV = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", imgHSV)
    hueMin = cv2.getTrackbarPos("Hue Min", "Trackbars")
    hueMax = cv2.getTrackbarPos("Hue Max", "Trackbars")
    satMin = cv2.getTrackbarPos("Sat Min", "Trackbars")
    satMax = cv2.getTrackbarPos("Sat Max", "Trackbars")
    valMin = cv2.getTrackbarPos("Val Min", "Trackbars")
    valMax = cv2.getTrackbarPos("Val Max", "Trackbars")
    lower = np.array([hueMin, satMin, valMin])
    upper = np.array([hueMax, satMax, valMax])
    mask = cv2.inRange(imgHSV, lower, upper)
    maskImg = stackImages(1, [[imageSmall, imgHSV, mask],
                            [imageSmall, imgHSV, mask]])
    cv2.imshow("output", maskImg)
    cv2.waitKey(1)