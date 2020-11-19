import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import face_extraction_dlib as fe


class FaceRecognition:
    #initialize required parameters
    def __init__(self, shape_predictor, model):
        self.shape_predictor = shape_predictor
        self.detector = dlib.get_frontal_face_detector()
        self.model = model
        self.class_names = []
        self.encoding_known_list = []
    
    #helper method to find distance between two encodings
    def find_euclidean_distance(self,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    #method to compare all known faces with the given images encodings 
    def face_distance(self ,encoding_face):
        distance = []
        for encoding in self.encoding_known_list:
            distance.append(self.find_euclidean_distance(encoding, encoding_face))
        return distance

    #method that generates boolean list containing the 
    def compare_faces(self, encoding_face):
        comp = []
        distance = self.face_distance( encoding_face)
        for d in distance:
            comp.append(not (d > 0.6))
        return comp

    #encode known faces
    def encodings_known_img(self,path="KnownFaces"):
        images = []
        mylist = os.listdir(path)
        print(mylist)

        for myclass in mylist:
            img = dlib.load_rgb_image(f'{path}/{myclass}')
            images.append(img)
            self.class_names.append(os.path.splitext(myclass)[0])
        encoding_list = []
        
        for img in images:    
            img_detect = fe.detect_faces(img, self.detector, self.shape_predictor)
            img_chip = img_detect[0][0]
            img_features = np.array(self.model.compute_face_descriptor(img_chip))
            encoding_list.append(img_features)
        self.encoding_known_list = encoding_list

    #uses all the above functions to get the match and add bounding boxes and names
    def face_recognition(self,img, scale_factor):
        img_small = cv2.resize(img,(0,0),None,scale_factor,scale_factor)
        img_detect = fe.detect_faces(img_small, self.detector, self.shape_predictor)
        left, right, top , bottom = 0,0,0,0
        cf_loacations = img_detect[2]
        cf_encoding =[]

        for i in img_detect[0]:
            print(i.shape)
            img_features = np.array(self.model.compute_face_descriptor(i))
            cf_encoding.append(img_features)

        rescale_factor = 1/scale_factor
        for encode_face , face_location in zip(cf_encoding, cf_loacations):
            face_dis = self.face_distance(encode_face)
            matches = self.compare_faces(encode_face)
            matchIndex = face_dis.index(min(face_dis))
            print(face_dis)
            print(matches)

            if matches[matchIndex]:
                name = self.class_names[matchIndex].upper()
                left = int(face_location.left()*rescale_factor)
                top = int(face_location.top()*rescale_factor)
                right = int(face_location.right()*rescale_factor)
                bottom = int(face_location.bottom()*rescale_factor)
                cv2.rectangle(img, (left, top), (right, bottom), (255,0,0),2)   
                cv2.rectangle(img, (left, bottom), (right, bottom+35), (255,0,0), cv2.FILLED)
                cv2.putText(img, name,(left+6, bottom+30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255, 255),2)
        return img
        
    #takes image as input and calls face_recognition method by passing image
    def show_image(self, path):
        img = cv2.imread(path)
        img = self.face_recognition(img, 2)
        img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        plt.imshow(img)
        plt.show()

    #takes video as input and calls face_recognition method on each frame
    def show_video(self, video):
        cap = cv2.VideoCapture(video)

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = self.face_recognition(img,0.5)
            cv2.imshow('Video Face Detect', img)
            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face Recognitoin")
    parser.add_argument("-i","--ivpath", type=str,
                        help="path of image or video to recognize face from.Enter 0 to read from primary camera")
    parser.add_argument("-s","--shape_predictor", type=str, help="path for shape_predictor_5_face_landmarks.dat file")
    parser.add_argument("-m","--model", type=str, help="path for dlib_face_recognition_resnet_model_v1.dat file")
    parser.add_argument("-ki","--known_images", type=str, help="path of folder containg known images for training/extracting encodings")
    args = parser.parse_args()

    
    if(args.shape_predictor == None):
        shape_predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")
    else:
        shape_predictor = dlib.shape_predictor(args.shape_predictor)
    if(args.model == None):
        model = dlib.face_recognition_model_v1("../data/dlib_face_recognition_resnet_model_v1.dat/dlib_face_recognition_resnet_model_v1.dat")
    else:
        model = dlib.face_recognition_model_v1(args.model)
    
    #Get FaceRecognition object
    frc = FaceRecognition( shape_predictor, model)

    if(args.known_images == None):
        frc.encodings_known_img()
    else:
        frc.encodings_known_img(args.known_images)
    print(frc.class_names)

    if(args.ivpath == None):
        print("File path not specified see help using -h")
    else:
        if args.ivpath.endswith(".jpg") or args.ivpath.endswith(".png") or args.ivpath.endswith(".jpeg"):
            frc.show_image(args.ivpath)
        elif args.ivpath.endswith(".mp4"):
            frc.show_video(args.ivpath)
        elif args.ivpath == '0':
            frc.show_video(0)
        else:
            print("File format not supported")
    