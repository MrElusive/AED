import cv

# This code was taken from http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html

# Load data for detecting faces
hc = cv.Load("haarcascade_frontalface_default.xml")
img = cv.LoadImage("../test_images/Happy.jpg", 0)

# Grab faces
faces = cv.HaarDetectObjects(img, hc, cv.CreateMemStorage())

# Draw the boundaries
for (x, y, w, h), n in faces:
    cv.Rectangle(img, (x, y), (x + w, y + h), 255)

cv.SaveImage("detected.jpg", img)
