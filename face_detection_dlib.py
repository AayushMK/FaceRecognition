#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy


import cv2
import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()


def detect_face(img):
    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for i, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     i, d.left(), d.top(), d.right(), d.bottom()))
        start = (d.left(), d.top())
        end = (d.right(), d.bottom())
        color = (255, 255, 255)
        cv2.rectangle(img, start, end, color, 10)
    return img


lisa_img = cv2.imread("./data/lisa.jpg")
chelis_img = cv2.imread("./data/chelis.jpg")

result = detect_face(chelis_img)

plt.imshow(result)
plt.show()

cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read(0)
    f = detect_face(frame)
    cv2.imshow('Video Face Detect', f)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
