import pandas as pd
import numpy as np
import math
import models.utils.learning_rate_functions as lr_functions
import models.utils.neighbourhood_functions as nh_functions
import random
class Kohonen:
    def __init__(self, logger, map_shape, max_epochs=300, learning_rate=0.1,
                learning_method=None, radius_method=None, **kwargs):
        '''
        Creates and initialize kohonen class network.
        Params:
        - logger: logging module compatuble class logger
        - map_shape (tuple): tuple of length 3 where each place represents (#_row, #_col, #_inputs)
        - max_epochs (int): max_number of epochs, this is used for linear eta decay
        - radius_method (str): can be one of ['exponential', 'linear', None]
        - learning_method (str): can be one of ['exponential', 'linear', None]
        '''
        self.logger = logger
        self.net = np.random.random(map_shape)
        self.max_epochs = max_epochs
        # set learning rate function and radius
        self.learning_rate_fun = self.select_learning_rate(learning_method, learning_rate, time_decay=kwargs.get('time_decay', 2))
        # note that this is a heuristic
        radius_val = max(map_shape[:-1]) / 2.0
        self.radius_fun = self.select_radius_decay_fun(radius_method, radius_val, time_decay=kwargs.get('time_decay', 2))
        # setup neighbourhood function
        self.neighbourhood_fun = self.select_neighbourhood_fun(self.radius_fun)


    def select_learning_rate(self, learning_method, learning_rate, **kwargs):
        '''
        Selects appropiate learning rate function and set the initial value.
        Params:
        - learning_method (string): selector to search for specific function.
        - learning_rate (float): initial value.

        Returns:
        - function with signature: (epoch)-> float
        '''
        if learning_method == 'exponential':
            time_decay = kwargs['time_decay']
            return lr_functions.exponential_decay(learning_rate, time_decay)

        if learning_method == 'linear':
            return lr_functions.linear_decay(learning_rate, self.max_epochs)

        if learning_method == None:
            # if none, use default ... a constant value
            return lr_functions.constant(learning_rate)

        raise NotImplementedError

    def select_radius_decay_fun(self, radius_method, radius_val,**kwargs):
        '''
        Selects appropiate radius method.
        Params:
        - radius_method (string): selector to search for specific function.
        Returns:
        - Function with signature: (epoch) -> float
        '''
        if radius_method == 'exponential':
            time_decay = kwargs['time_decay']
            return nh_functions.exponential_decay(radius_val, time_decay)

        if radius_method == 'linear' or radius_method == None:
            return nh_functions.linear_decay(radius_val, self.max_epochs)

        raise NotImplementedError

    def select_neighbourhood_fun(self, radius_func):
        '''
        Returns the function construction based on the radius function to use
        Pramas:
        - radius_func: function with signature (distance, epoch) -> float
        '''
        return lambda d, epoch: math.exp((- d *2)/(2 * radius_func(epoch)**2))

    def fit(self, input_data):
        '''
        Train a model based on the input information
        Params:
        - input_data: numpy array of shape (none, #_attributes)
        '''
        for epoch in range(self.max_epochs):
            # loop over the input data
            self.logger.info('[KOHONEN] training epoch {} ...'.format(epoch))
            for sample_vector in input_data:
                # take a sample
                # sample_vector = random.choice(input_data.tolist())
                # sample_vector = np.array(sample_vector)
                # returns a list of tuples consisting of :
                # (eucledean_distance, diference_vector, (row, col))

                distances = self._caculate_distances(sample_vector, self.net)
                # find best matching unit
                bmu = min(distances, key=lambda tuple: tuple[0])
                # find bmu's neighbours
                current_radius = self.radius_fun(epoch)
                neighbours = self._find_neighbours(bmu[2], distances, current_radius)
                # update neighbours
                self.net = self._update_net(self.net, neighbours, epoch)

        self.logger.info('[Kohonen]: TRAINING COMPLETE! epochs: {}'.format(self.max_epochs))

    def _caculate_distances(self, input_vec, net):
        '''
        Calculates the eucledean distance of each node in the feature map to the input vec.
        Params:
        - input_vec (numpy array): input vector of size (1, K-attributes)
        - net (numpy array): wight network of size (rows cols, inputs)

        Retunrs:
        - list of tuples (distance, distance_vec, coordinates_in_net)
        '''
        distances = []
        for row in range(net.shape[0]):
            for col in range(net.shape[1]):
                distance_vec = net[row, col, :] - input_vec
                distance = np.linalg.norm(distance_vec)
                distances.append((distance, distance_vec, (row, col)))

        return distances

    def _find_neighbours(self, bmu, distances, current_radius):
        '''
        find neighbours to position in the map based on the radius.
        Params:
        - bmu (tuple): coordinates of the winner unit.
        - distances (list of tuples): results of passing an input thorugh the net.
        - current_radius (float): value representing the max distance a unit can be away from the winner

        Returns:
        - list of coordinates of all neighbours.

        Obs:
        THIS METHOD WILL INCLUDE THE BMU AS A NEIGHBOUR
        '''
        neighbours = []
        for neighbour in distances:
            radius = np.linalg.norm(np.array(bmu) - np.array(neighbour[2]))

            # check that is near enough
            if radius > current_radius:
                break
            # now we replace neighour tuple's first value of distance with the radius to the
            # bmu
            neighbour = (radius, neighbour[1], neighbour[2])
            neighbours.append(neighbour)

        return neighbours

    def _update_net(self, net, neighbours, epoch):
        '''
        Updates all units inside neighbours list
        Params:
        - net (numpy array of 3-dimensions): weight representing connections between feature map
            and input
        - neighbours (list of tuples): representing the output of finding the bmu neighbours
        '''
        for neighbour in neighbours:
            # get current learning rate
            lr = self.learning_rate_fun(epoch)
            # get current neighbour regularization
            beta = self.neighbourhood_fun(neighbour[0], epoch)
            delta_weight = lr * beta * neighbour[1]
            row, col = neighbour[2]
            net[row, col, :] += delta_weight
        return net

    def predict(self, input_data):
        '''
        returns the unit coordinates of the bmu
        Params:
        - input_data (numpy array)L input array of shape (None, K-params)

        Returns:
        - list of coordinates for the bmu of each input
        '''
        predictions = []
        for row in input_data:
            distances = self._caculate_distances(row, self.net)
            bmu = min(distances, key=lambda tuple: tuple[0])
            predictions.append(bmu[2])
        return predictions
