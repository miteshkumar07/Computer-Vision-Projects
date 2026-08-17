import numpy as np
import cv2


class NearestNeighborClassifier:

    def __init__(self):
        self.classifier = cv2.ml.KNearest_create()
        self.k_neighbours=1
        self.__reset()

    def __reset(self):
        self.classifier.setDefaultK(1)

    def fit(self, embeddings, labels):
        self.__reset()
        self.classifier.train(embeddings.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32))
        
    def predict_labels_and_similarities(self, embeddings):
        _, prediction_labels,vector_result, dists = self.classifier.findNearest(embeddings.astype(np.float32),self.k_neighbours)
        similarities = dists.flatten()
        return prediction_labels.flatten(),vector_result, similarities
    
    # For setting the number of neighbours
    def set_k_neighbours(self,k):
        self.k_neighbours=k