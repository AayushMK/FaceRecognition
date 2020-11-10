# This program straightens the faces in the image, extracts them and saves them.
# MTCNN is used for detecting detect faces

from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


detector = MTCNN()


def detect_faces(img):
    faces = detector.detect_faces(img)

    for x in faces:
        start = (x['box'][0], x['box'][1])
        end = (x['box'][0]+x['box'][2], x['box'][1] + x['box'][3])
        color = (255, 255, 255)
        cv2.rectangle(img, start, end, color, 10)

        p1 = x['keypoints']['left_eye']
        p2 = x['keypoints']['right_eye']

        cv2.circle(img, p1, 20, (255, 255, 0), -1)
        cv2.circle(img, p2, 20, (255, 255, 0), -1)

    return img


img = cv2.cvtColor(cv2.imread("./data/david.jpg"), cv2.COLOR_BGR2RGB)
detected_img = detect_faces(img)
plt.imshow(detected_img)
plt.show()

cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read(0)
    frame = detect_faces(cv2.flip(frame, 1))
    cv2.imshow('Video Face Detect', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
