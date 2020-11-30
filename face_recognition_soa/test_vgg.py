from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from numpy import asarray
from matplotlib import pyplot
import cv2

class evluate_model():
    def __init__(self):
        self.flp = fetch_lfw_pairs(subset='test', color=True, resize=1  # this transform inputs to (125, 94) from (62, 47)
                                        )
#helper fucntion to get embeddings of list of images
    def get_embeddings(self, samples):

        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)
        # create a vggface model
        model = VGGFace(model='resnet50', include_top=False,
                        input_shape=(224, 224, 3), pooling='avg')
        # perform prediction

        yhat = model.predict(samples)
        return yhat

#helper function to calculate distance and predict
    def is_match(self, known_embedding, candidate_embedding, thresh=0.5):
        # calculate distance between embeddings
        score = cosine(known_embedding, candidate_embedding)
        obj = {}
        obj["distance"] = score
        if score <= thresh:
            obj["verified"] = True
        else:
            obj["verified"] = False
        return obj

#makes images ready for evaluation
    def preprocess_data(self):
        pairs = self.flp.pairs
        instances = pairs[0]
        pairs_first_column = pairs[:, 0]
        pairs_second_column = pairs[:, 1]
        all_faces = []
        print("resizing")
        for i in range(0, 1000):
            all_faces.append(cv2.resize(
                pairs_first_column[i], (224, 224), interpolation=cv2.INTER_AREA))
        for i in range(0, 1000):
            all_faces.append(cv2.resize(
                pairs_second_column[i], (224, 224), interpolation=cv2.INTER_AREA))
        print("resiziing finished")
        return all_faces

#get embeddings and prediction for all the faces 
    def evaluate(self, all_faces):
        actuals = []
        predictions = []
        distances = []

        print("embedding started")
        embeddings = self.get_embeddings(all_faces)
        print("embedding finished")
        
        embeddings_first = embeddings[0:1000]
        embeddings_second = embeddings[1000:2000]

        labels = self.flp.target
        target_names = self.flp.target_names
        pairs = self.flp.pairs
        instances = pairs[0]
        for i in range(1000):
            face = []
            pair = pairs[i]
            obj = self.is_match(embeddings_first[i], embeddings_second[i], thresh=0.5)
            prediction = obj["verified"]
            predictions.append(prediction)

            distances.append(obj["distance"])

            label = target_names[labels[i]]
            actual = True if labels[i] == 1 else False
            actuals.append(actual)
        return actuals, predictions

#cacluates different metrics
    def show_metric(self, actuals, predictions):
        accuracy = 100*accuracy_score(actuals, predictions)
        precision = 100*precision_score(actuals, predictions)
        recall = 100*recall_score(actuals, predictions)
        f1 = 100*f1_score(actuals, predictions)

        print("instances: ", len(actuals))
        print("accuracy: ", accuracy, "%")
        print("precision: ", precision, "%")
        print("recall: ", recall, "%")
        print("f1: ", f1, "%")

        cm = confusion_matrix(actuals, predictions)

        tn, fp, fn, tp = cm.ravel()
        print((tn, fp, fn, tp))

if __name__ == "__main__":
    em = evluate_model()
    all_faces = em.preprocess_data()
    actuals, predictions = em.evaluate(all_faces)
    em.show_metric(actuals, predictions)
