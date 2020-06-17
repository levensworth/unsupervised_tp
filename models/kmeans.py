import numpy as np
import pandas as pd

import random
# this is a basic implementation fo k_means algorithm

class KMeans():
    def __init__(self, n_classes, logger=None, init_method='random', max_iter=10):
        # set logger object, should be logging protocol compliant
        self.logger = logger
        self.n_classes = n_classes
        self.max_iter = max_iter
        # initialize centroids based on init_method
        self.init_method = init_method

    def fit(self, train_data, train_labels):
        '''
        train model based on the init method and the train data.
        Will iterate over dataset for [max_iters]
        if logger provided => will output iteration values
        Params:
        - train_data: numpy array of (n_obs, variables)
        - train_labels: list of labels of length n_obs
        '''
        possible_labels = set(train_labels)
        self.space_dim = train_data.shape[1]
        classification = []
        self.centroids = []
        # init labels
        for entry in train_data:
            label = random.choice(list(possible_labels))
            classification.append((entry, label))

        # interate util convergance or max iterations
        epoch = 0
        prev_classification = None
        while not self._check_converge(prev_classification, classification) and epoch < self.max_iter:
            self.logger.info('[Training] epoch: {}'.format(epoch))
            prev_classification = classification.copy()
            classification = []

            for label in possible_labels:
                centroid = self._calculate_centroid(prev_classification, label)
                self.centroids.append((label, centroid))
            for entry in train_data:
                new_label = self._classify_obs(entry, self.centroids)
                classification.append((entry, new_label))
            epoch += 1

        self.logger.info('[Training] Done training, enjoy model predicitons!')

    def predict(self, data):
        '''
        Predict classification for dataset
        Params:
        - data: numpy array of shape (None, space_dim). Where space_dim is the same dim as training data

        Returns:
        - numpy array with given classificaitons in same order as data
        '''
        predictions = []
        for entry in data:
            pred = self._classify_obs(entry, self.centroids)
            predictions.append(pred)

        return np.array(predictions)

    def _calculate_centroid(self, observations, label):
        '''
        calculate the centroid for a label given observations
        Params:
        - observations: list of tuples (entry, label)
        - label: the current label for which a centroid should be calcualted

        Returns:
        - a list of size len(entry) representing a coordinate in the given space
        '''
        # centroid is the mean value of all points
        centroid = np.zeros(self.space_dim)
        amount_of_entries = 0

        for entry, current_label in observations:
            if current_label != label:
                break

            centroid += entry
            amount_of_entries += 1
        if amount_of_entries > 0:
            centroid /= amount_of_entries
        return centroid

    def _classify_obs(self, entry, centroids):
        '''
        Classify an entry given all centroid by least ecludean distance to each one
        Params:
        - entry: array like object
        - centroids: list of tuples (label, point). Where point is an array
            like objects representing points in the same space as entry

        Returns
        - entroid
        '''
        disntaces_tuples = []
        for label, point in centroids:
            # numpy norm is l2, that's why this is the same as eucledean distance
            distance = np.linalg.norm(entry - point)
            disntaces_tuples.append((distance, label))
        min_tuple = min(disntaces_tuples, key = lambda t: t[0])
        return min_tuple[1]


    def _check_converge(self, prev_classification, new_classification):
        '''
        Checks if both classification are equals
        Params:
        - prev_classification: list of tuples (entry, label)
        - new_classification: list of tuples (entry, label)
        Rturns:
        - bool: true if both lists are equals flase if not
        '''
        if prev_classification == None or new_classification == None:
            return False

        for idx in range(len(prev_classification)):
            entry, prev_label = prev_classification[idx]
            entry, new_label = new_classification[idx]
            if prev_label != new_label:
                return False

        return True
