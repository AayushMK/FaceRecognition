import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def face_distance(encoding_known_list, encoding_face):
    distance = []
    for encoding in encoding_known_list:
        distance.append(findEuclideanDistance(encoding, encoding_face))
    return distance

def compare_faces(encoding_known_list, encoding_face):
    comp = []
    distance = face_distance(encoding_known_list, encoding_face)
    for d in distance:
        comp.append(not (d > 0.6))
    return comp
    
def find_encodings(images):
    encoding_list = []
    for img in images:
        img_detect = detector(img, 1)
        
        img_shape = shape_predictor(img, img_detect[0])
        
        img_chip = dlib.get_face_chip(img, img_shape)
        # plt.imshow(img_chip)
        # plt.show()
        img_features = np.array(model.compute_face_descriptor(img_chip))
        encoding_list.append(img_features)
    return encoding_list


def face_recognition(img, scale_factor):
    img_small = cv2.resize(img,(0,0),None,scale_factor,scale_factor)
    img_small = cv2.cvtColor(img_small,cv2.COLOR_BGR2RGB)
    img_detect = detector(img_small, 1)
    print(img_detect)
    left, right, top , bottom = 0,0,0,0
    cf_loacations = []
    cf_encoding = []
    for k , d in enumerate(img_detect ):
        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()
        cf_loacations.append(d)
        print(left)
        img_shape = shape_predictor(img_small, d)
        img_chip = dlib.get_face_chip(img_small, img_shape)
        img_features = np.array(model.compute_face_descriptor(img_chip))
        cf_encoding.append(img_features)
    
    rescale_factor = 1/scale_factor
    print(rescale_factor)
    for encode_face , face_location in zip(cf_encoding, cf_loacations):
        face_dis = face_distance(encoding_known_list,encode_face)
        matches = compare_faces(encoding_known_list,encode_face)
        matchIndex = face_dis.index(min(face_dis))
        print(face_dis)
        print(matches)

        if matches[matchIndex]:
            name = class_names[matchIndex].upper()
            left = int(face_location.left()*rescale_factor)
            top = int(face_location.top()*rescale_factor)
            right = int(face_location.right()*rescale_factor)
            bottom = int(face_location.bottom()*rescale_factor)
            cv2.rectangle(img, (left, top), (right, bottom), (255,0,0),2)   
            cv2.rectangle(img, (left, bottom), (right, bottom+35), (255,0,0), cv2.FILLED)
            cv2.putText(img, name,(left+6, bottom+30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255, 255),2)
    return img

def show_image(path):
    img = cv2.imread(path)
    img = face_recognition(img, 2)
    img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.show()

def show_video(video):
    cap = cv2.VideoCapture(video)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = face_recognition(img,0.5)
        cv2.imshow('Video Face Detect', img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face Recognitoin")
    parser.add_argument("path", type=str,
                        help="path of image or video to recognize face from.Enter 0 to read from primary camera")
    args = parser.parse_args()


    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("../data/shape_predictor_5_face_landmarks.dat/shape_predictor_5_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("../data/dlib_face_recognition_resnet_model_v1.dat/dlib_face_recognition_resnet_model_v1.dat")
    path = "KnownFaces"
    images = []
    class_names = []
    mylist = os.listdir(path)
    print(mylist)

    for myclass in mylist:
        img = dlib.load_rgb_image(f'{path}/{myclass}')
        images.append(img)
        class_names.append(os.path.splitext(myclass)[0])

    print(class_names)

    encoding_known_list = find_encodings(images)

    print(len(encoding_known_list))


    

    if args.path.endswith(".jpg") or args.path.endswith(".png") or args.path.endswith(".jpeg"):
        show_image(args.path)
    elif args.path.endswith(".mp4"):
        show_video(args.path)
    elif args.path == '0':
        show_video(0)
    else:
        print("File format not supported")
    