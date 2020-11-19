# This program straightens the faces in the image, extracts them and displays them.
# dlib is used for detecting detect faces

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import dlib

class FaceExtractionDlib:
    def __init__(self, shape_predictor="../data/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.face_landmarks = dlib.shape_predictor(shape_predictor)

    def getAngle(self, p1, p2):
        slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        angle = math.atan(slope)*(180/3.14)
        return angle


    def detect_faces(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
        img_for_crop = img.copy()
        all_faces = []
        full_bgr_img = img.copy()
        locations = []
        if len(faces) > 0:
            for i, x in enumerate(faces):
                x1 = x.left()
                y1 = x.top()
                x2 = x.right()
                y2 = x.bottom()
                start = (x1, y1)
                end = (x2,y2)
                locations.append(x)
                color = (255, 255, 255)
                cv2.rectangle(full_bgr_img, start, end, color, 5)
                landmarks = self.face_landmarks(gray_img, x)
                elx1=landmarks.part(36).x 
                elx2=landmarks.part(39).x 
                ely1=landmarks.part(37).y 
                ely2=landmarks.part(40).y

                erx1=landmarks.part(42).x 
                erx2=landmarks.part(45).x 
                ery1=landmarks.part(43).y 
                ery2=landmarks.part(46).y 

                #get center points of eyes
                center1_x = int((elx1+elx2)/2)
                center1_y = int((ely1+ely2)/2)
                center2_x = int((erx1+erx2)/2)
                center2_y = int((ery1+ery2)/2)

                p1 = ((elx1+elx2)/2 ,(ely1+ely2)/2)
                p2 = ((erx1+erx2)/2 ,(ery1+ery2)/2)

                cv2.circle(full_bgr_img, (center1_x, center1_y), 5, (255, 255, 0), -1)
                cv2.circle(full_bgr_img, (center2_x, center2_y), 5, (255, 255, 0), -1)

                angle = self.getAngle(p1, p2)             # get angle

                # create rotation matrix at origin point p1 and angle= angle
                matrix = cv2.getRotationMatrix2D(p1, angle, 1)

                # rotate image to get new rotated image
                new_image = cv2.warpAffine(
                    img_for_crop, matrix, (img_for_crop.shape[1], img_for_crop.shape[0]))


                # crop new image using old dimensions
                crop = new_image[y1:y2, x1:x2]
                crop = cv2.resize(crop, (150, 150),interpolation=cv2.INTER_AREA)
            
                if crop.size is not 0:
                    # convert to bgr to save in right format
                    # bgr_img = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    all_faces.append(crop)

        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(full_bgr_img, 'No face detected',  (50, 50),
                        font, 1, (0, 255, 255),  2,  cv2.LINE_4)
        return [all_faces, full_bgr_img, locations]


if __name__ == '__main__':
    fe = FaceExtractionDlib() 
    cap = cv2.VideoCapture(0)

    while True:
        rect, frame = cap.read(0)
        # my_frame = cv2.resize(frame,None, fx=0.50, fy=0.5)
        my_frame = cv2.flip(frame, 1)
        f = fe.detect_faces(my_frame)

        cv2.imshow("Full video", f[1])

        # append faces
        if len(f[0]) > 0:
            faces = []
            for face in f[0]:
                faces.append(face)
            all_faces = np.hstack(faces)
            cv2.imshow("Faces", all_faces)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
