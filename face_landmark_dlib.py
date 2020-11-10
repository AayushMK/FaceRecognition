import dlib
import cv2
import matplotlib.pyplot as plt

hog_face_detector = dlib.get_frontal_face_detector()
face_landmark = dlib.shape_predictor(
    "./data/shape_predictor_68_face_landmarks.dat")


def draw_landmark(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray_img)
    for face in faces:
        landmarks = face_landmark(gray_img, face)
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x, y), 1, (255, 255, 0), 1)
    return img


lisa_img = cv2.imread("./data/lisa.jpg")
chelis_img = cv2.imread("./data/chelis.jpg")
img = draw_landmark(chelis_img)
plt.imshow(img)
plt.show()
cap = cv2.VideoCapture(0)

while True:
    rect, frame = cap.read(0)
    f = draw_landmark(cv2.flip(frame, 1))
    cv2.imshow('Video Face Detect', f)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
