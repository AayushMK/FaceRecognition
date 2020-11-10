# This program straightens the faces in the image, extracts them and saves them.
# MTCNN is used for detecting detect faces

from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


detector = MTCNN()


def getAngle(p1, p2):

    slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    angle = math.atan(slope)*(180/3.14)
    print("angle ="+str(angle))
    return angle


def saveImg(img, name):
    print(cv2.imwrite("data/faces/"+name+".jpg", img))


def detect_faces(img, name):
    faces = detector.detect_faces(img)
    img_for_crop = img.copy()
    for x in faces:
        start = (x['box'][0], x['box'][1])
        end = (x['box'][0]+x['box'][2], x['box'][1] + x['box'][3])
        color = (255, 255, 255)
        cv2.rectangle(img, start, end, color, 10)

        p1 = x['keypoints']['left_eye']
        p2 = x['keypoints']['right_eye']
        print(p1)
        print(p2)
        cv2.circle(img, p1, 20, (255, 255, 0), -1)
        cv2.circle(img, p2, 20, (255, 255, 0), -1)

        px = x['box'][0]                     # point x
        py = x['box'][1]                     # point y
        pa = x['box'][0]+x['box'][2]         # x + width
        pb = x['box'][1] + x['box'][3]       # y + height

        angle = getAngle(p1, p2)             # get angle

        # create rotation matrix at origin point p1 and angle= angle
        matrix = cv2.getRotationMatrix2D(p1, angle, 1)

        # rotate image to get new rotated image
        new_image = cv2.warpAffine(
            img_for_crop, matrix, (img_for_crop.shape[1], img_for_crop.shape[0]))

        # crop new image using old dimensions
        crop = new_image[py:pb, px:pa]

        # convert to bgr to save in right format
        bgr_img = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # for unique names
        full_name = name + str(px)
        saveImg(bgr_img, full_name)

    plt.imshow(img)
    plt.show()


img = cv2.cvtColor(cv2.imread("./data/david.jpg"), cv2.COLOR_BGR2RGB)
detect_faces(img, "david")


# The detector outputs the following data
# [
#     {
#         'box': [277, 90, 48, 63], #x,y,width,height
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]
