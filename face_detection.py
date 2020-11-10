import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(
    "./data/haarcascade_frontalface_default.xml")


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(
        face_img, scaleFactor=1.2, minNeighbors=5)

    for(x, y, w, h) in face_rects:
        start = (x, y)
        end = (x+w, y+h)
        color = (255, 255, 255)
        cv2.rectangle(face_img, start, end, color, 10)
    return face_img


lisa_img = cv2.imread("./data/lisa.jpg")
chelis_img = cv2.imread("./data/chelis.jpg")

result = detect_face(lisa_img)

plt.imshow(result)
plt.show()

cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read(0)
    frame = detect_face(cv2.flip(frame, 1))
    cv2.imshow('Video Face Detect', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
