import cv2
import numpy as np
import pickle
import os
import random
from sklearn.cluster import KMeans
from face_detector import FaceDetector

import matplotlib.pyplot as plt

from classifier import NearestNeighborClassifier

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))
        self.classifier= NearestNeighborClassifier()
        self.label_dict = {"Alan Ball":0,"Manuel Pellegrini":1,"Marina Silva":2,"Nancy Sinatra":3,"Peter Gilmour":4}

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)
            
    # Update the embeddings accordingly
    def update(self, face, label):
        self.embeddings = np.append(self.embeddings, [self.facenet.predict(face)], axis=0) 
         
        self.labels.append(self.label_dict[label])
        #self.nearest_neighbor_classifier.fit(self.embeddings, np.array(self.labels))
        self.classifier.fit(self.embeddings,np.array(self.labels))
        self.save()
        
    # Get the embeddings and predict
    def predict(self, face):
        
        # Get the embeddings
        embeddings = self.facenet.predict(face)
        embeddings = embeddings.reshape(1,-1)
        
        # Fit the classifier
        labels = np.array(self.labels)
        # Get the unique labels	
        unique_labels = list(set(labels))
        self.classifier.fit(self.embeddings, labels)
        
        # Predict the label
        posterior_probabilities = [0] * len(set(self.labels))
        self.classifier.set_k_neighbours(self.num_neighbours)
        predictions,vector_result,similiarities = self.classifier.predict_labels_and_similarities(embeddings)
        
        predicted_label = int(predictions[0])
        
        for i, class_label in enumerate(unique_labels):
            # Get the number of neighbours with the same label
            ki = np.count_nonzero(vector_result == class_label)
            
            # Calculate the posterior probabilities for each class
            posterior_probabilities[i] = ki / self.num_neighbours
            
        posterior_prob = posterior_probabilities[predicted_label]
        distance_to_class = min(similiarities.flatten())
        
        # Get the predicted person
        predicted_person = get_key_from_value(self.label_dict,predicted_label)
        
        # For Open Set Identification
        if posterior_prob < self.min_prob or distance_to_class > self.max_distance:
            predicted_person = "Unknown"
        return predicted_person, posterior_prob, distance_to_class

# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=5, max_iter=25):
        # Prepare FaceNet.
        self.facenet = FaceNet()
        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_centers = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []
        self.k_mean_inertia=[]
        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_centers, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_centers, self.cluster_membership) = pickle.load(f)

    # Update the embedding and fit the model later
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        self.fit()
        #print(self.cluster_membership)
        self.save()
        
    # Implement k-means clustering 
    def fit(self):
        # Initialize cluster centers randomly
        self.cluster_centers = self.embeddings[np.random.choice(len(self.embeddings), self.num_clusters, replace=True)]

        k_means_objective_values = []
        for _ in range(self.max_iter):
            # Assign each sample to the nearest cluster
            distances = np.linalg.norm(self.embeddings[:, np.newaxis] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.array([self.embeddings[labels == i].mean(axis=0) for i in range(self.num_clusters)])

            # Check for convergence
            if np.allclose(new_centers, self.cluster_centers, rtol=1e-4):
                break

            self.cluster_centers = new_centers

            # Calculate k-means inertia (sum of squared distances to nearest cluster center)
            inertia = np.sum(np.min(distances, axis=1))
            k_means_objective_values.append(inertia)
            
        #print(labels)
        self.cluster_membership = labels
        self.plot_k_means_objective_values(k_means_objective_values)

    def plot_k_means_objective_values(self,k_means_objective_values):
        plt.plot(range(1, len(k_means_objective_values) + 1), k_means_objective_values, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('K-means Objective Function Value')
        plt.title('K-means Objective Function Value Over Iterations')
        plt.show()
    
    # Predict the closest embedding which resembles the face in the cluster arrangement
    def predict(self, face):
        # Assign the face to the nearest cluster
        embedding = self.facenet.predict(face).reshape(1, -1)
        distances = np.linalg.norm(embedding - self.cluster_centers, axis=1)
        
        # Return the cluster index and the distance to the cluster center
        predicted_cluster = np.argmin(distances)
        minimum_distance = np.min(distances)
        
        return predicted_cluster,minimum_distance

# To get the the name of the person from the label
def get_key_from_value(dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None
   
