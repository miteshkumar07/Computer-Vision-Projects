import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        
        similarity_thresholds = []
        identification_rates = []
        # Train the classifier on the given training data.
        self.classifier.fit(self.train_embeddings, self.train_labels)
        print("Hi")
        # Predict similarities for the given test data.
        #self.classifier.set_k_neighbours(11)
        prediction_labels,_, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        #print(prediction_labels.shape)
        similarities = -1 * similarities
        # Calculate identification rates for different similarity thresholds.
        for false_alarm_rate in self.false_alarm_rate_range:
            similarity_threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            #print(similarity_threshold)
            
            unknown_index = np.where(similarities <= similarity_threshold)
            new_prediction_labels = prediction_labels.copy()
            #print(unknown_index)
            
            # Set all predictions with similarity below the threshold to the unknown label.
            new_prediction_labels[unknown_index] = UNKNOWN_LABEL
            identification_rate = self.calc_identification_rate(new_prediction_labels)

            # Store the results.
            similarity_thresholds.append(similarity_threshold)
            identification_rates.append(identification_rate)
            
        evaluation_results = {'similarity_thresholds': similarity_thresholds,'identification_rates': identification_rates}
        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):

        # Find the similarity threshold for the given false alarm rate.
        unknowns_labels = np.argwhere(self.test_labels == UNKNOWN_LABEL)
        unknowns = similarity[unknowns_labels]
        
        # For finding threshold.
        threshold = np.percentile(unknowns, (1-false_alarm_rate)*100)
        
        return threshold 

    def calc_identification_rate(self, prediction_labels):
        # num of true identification
        count_true_id = np.logical_and(self.test_labels != UNKNOWN_LABEL, prediction_labels == self.test_labels)
        sum_true_id = np.sum(count_true_id)
        
        # num of labels which are not unknown
        normalize = np.sum(self.test_labels != UNKNOWN_LABEL)

        return sum_true_id/normalize
