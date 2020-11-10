from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.cvtColor(cv2.imread("./data/david.jpg"), cv2.COLOR_BGR2RGB)
img_copy = img.copy()
detector = MTCNN()
x = detector.detect_faces(img)

start = (x[0]['box'][0], x[0]['box'][1])
end = (x[0]['box'][0]+x[0]['box'][2], x[0]['box'][1] + x[0]['box'][3])
color = (255, 255, 255)
cv2.rectangle(img, start, end, color, 10)

p1 = x[0]['keypoints']['left_eye']
p2 = x[0]['keypoints']['right_eye']
print(p1)
print(p2)
cv2.circle(img, p1, 20, (255, 255, 0), -1)
cv2.circle(img, p2, 20, (255, 255, 0), -1)
plt.imshow(img)
plt.show()
# cv2.imshow("output", img)
# cv2.waitKey(0)


def getAngle(p1, p2):

    slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    angle = math.atan(slope)*(180/3.14)
    print(angle)
    return angle


angle = getAngle(p1, p2)
img2 = img_copy

matrix = cv2.getRotationMatrix2D(p1, angle, 1)
new_image = cv2.warpAffine(img2, matrix, (img2.shape[1], img2.shape[0]))
cv2.rectangle(new_image, start, end, color, 10)
plt.imshow(new_image)
plt.show()
