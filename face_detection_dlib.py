
import cv2
import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()


def detect_face(img):
    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(face_img, 1)

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
    f = detect_face(cv2.flip(frame, 1))
    cv2.imshow('Video Face Detect', f)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
