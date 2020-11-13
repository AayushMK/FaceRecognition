# This program straightens the faces in the image, extracts them and displays them.
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


def detect_faces(img):

    faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_for_crop = img.copy()
    all_faces = []
    full_bgr_img = img.copy()
    if len(faces) > 0:
        for x in faces:
            start = (x['box'][0], x['box'][1])
            end = (x['box'][0]+x['box'][2], x['box'][1] + x['box'][3])
            color = (255, 255, 255)
            cv2.rectangle(full_bgr_img, start, end, color, 5)

            p1 = x['keypoints']['left_eye']
            p2 = x['keypoints']['right_eye']
            print(p1)
            print(p2)
            cv2.circle(full_bgr_img, p1, 5, (255, 255, 0), -1)
            cv2.circle(full_bgr_img, p2, 5, (255, 255, 0), -1)

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
            print(new_image.shape)

            # crop new image using old dimensions
            crop = new_image[py:pb, px:pa]

            print(crop)
            if crop.size is not 0:
                # convert to bgr to save in right format
                # bgr_img = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                all_faces.append(crop)

    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(full_bgr_img, 'No face detected',  (50, 50),
                    font, 1, (0, 255, 255),  2,  cv2.LINE_4)
    return [all_faces, full_bgr_img]


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while True:
        rect, frame = cap.read(0)
        # my_frame = cv2.resize(frame,None, fx=0.50, fy=0.5)
        my_frame = cv2.flip(frame, 1)
        f = detect_faces(my_frame)

        cv2.imshow("Full video", f[1])

        # append faces
        if len(f[0]) > 0:
            faces = []
            for face in f[0]:
                faces.append(cv2.resize(face, (200, 280),
                                        interpolation=cv2.INTER_AREA))
            all_faces = np.hstack(faces)
            cv2.imshow("Faces", all_faces)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
