import pandas as pd
import numpy as np
import itertools
from heapq import heappush, heappop, heapify
import statistics
import math
import logging
from internal_cluster import AbstractNode, TerminalNode

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt



class HierarchyTree():
    def __init__(self, logger, mode=None):
        self.logger = logger
        # strategy for cluster distance measurement
        self.mode = mode
        self.root = None

    def fit(self, data):
        '''
        Train model based on the input dataset.
        Params:
        - data: numpy array of shape (None, dim)
        '''
        # we initialize the algorithm with each entry as a dif group
        # groups is a list of tuples -> each tuples consist of (cluster center, [items])
        groups = []
        space_dim = data.shape[1]
        for row in data:
            groups.append(TerminalNode(row))

        group_distances = []
        # by default this is a max_heap
        heapify(group_distances)

        while len(groups) > 1:
            group_distances = []
            # by default this is a max_heap
            heapify(group_distances)

            for node_1, node_2 in itertools.combinations(groups,2):
                # search through all possible combinations
                distance = self._calculate_cluster_distance(node_1, node_2)
                # add distance triplet to heap
                heappush(group_distances, (distance, node_1, node_2))

            distance, node_1, node_2 = heappop(group_distances)
            self.logger.info('grouping clusters with disatnce {}'.format(distance))

            # create new node
            new_group = AbstractNode()
            new_group.add_sub_cluster(node_1)
            new_group.add_sub_cluster(node_2)
            # remove groups from existing grups
            groups.remove(node_1)
            groups.remove(node_2)
            # add new group
            groups.append(new_group)

        self.root = groups.pop()

    def _calculate_cluster_distance(self, cluster_1, cluster_2):
        '''
        Calculate the distance between 2 clusters
        returns a float
        '''
        if self.mode == 'max':
            return self._max_cluster_distance(cluster_1, cluster_2)
        if self.mode == 'min':
            return self._min_cluster_distance(cluster_1, cluster_2)
        if self.mode == 'centroid':
            return self._centroid_distance(cluster_1, cluster_2)
        if self.mode == 'avg':
            return self._avg_cluster_distance(cluster_1, cluster_2)
        # if we are here it means the user specified an invalid strategy
        self.logger.error('No {} startegy implementation found'.format(self.mode))
        raise NotImplementedError

    def _max_cluster_distance(self, cluster_1, cluster_2):
        '''
        Calculate distance between cluster_1 and cluster_2 as the max distance bewteen their
        respective members.
        Params:
        - cluster_1: Cluster
        - cluster_2: Cluster
        '''

        max_distance = -1
        for item in cluster_1.get_sub_clusters():
            # calculate distance between item and each member of the other cluster
            if isinstance(item, TerminalNode):
                max_distance = max(max_distance, item.get_max_distance(cluster_2))
            else:
                sub_distance = self._max_cluster_distance(item, cluster_2)
                max_distance = max(max_distance, sub_distance)

        return max_distance

    def _min_cluster_distance(self, cluster_1, cluster_2):
        '''
        Calculate distance between cluster_1 and cluster_2 as the min distance bewteen their
        respective members.
        Params:
        - cluster_1: Cluster
        - cluster_2: Cluster
        '''
        min_distance = math.inf
        for item in cluster_1.get_sub_clusters():
            # calculate distance between item and each member of the other cluster
            if isinstance(item, TerminalNode):
                min_distance = min(min_distance, item.get_min_distance(cluster_2))
            else:
                sub_distance = self._min_cluster_distance(item, cluster_2)
                min_distance = min(min_distance, sub_distance)

        return min_distance

    def _centroid_distance(self, cluster_1, cluster_2):
        '''
        Calculate distance between cluster_1 and cluster_2 as the center mass distance bewteen their
        respective centroid.
        Params:
        - cluster_1: Cluster
        - cluster_2: Cluster
        '''
        centroid_1 = cluster_1.get_centroid()
        centroid_2 = cluster_2.get_centroid()
        return np.linalg.norm(centroid_1 - centroid_2)

    def _avg_cluster_distance(self, cluster_1, cluster_2):
        '''
        Calculate distance between cluster_1 and cluster_2 as the avg distance bewteen their
        respective member.
        Params:
        - cluster_1: Cluster
        - cluster_2: Cluster
        '''
        distances = []
        for item in cluster_1.get_sub_clusters():
            # calculate distance between item and each member of the other cluster
            if isinstance(item, TerminalNode):
                distances += item.get_all_distance(cluster_2)
            else:
                distances += self._avg_cluster_distance(item, cluster_2)
        if len(distances) == 0:
            return 0

        return sum(distances) / len(distances)

if __name__ == '__main__':
    # X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
    X = np.array([
        [1, 1],
        [3, 3],
        [0, 0],
        [5.6, 5.6]
    ])

    # TODO: numpy is empty no devuelve un array vacio ... las validaciones no funcionan


    # Z = linkage(points, 'single')
    # fig = plt.figure()
    # dn = dendrogram(Z)
    # plt.show()

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    model = HierarchyTree(logging, 'avg')
    model.fit(X)
    print('done')

# primero : (1, 2)=> (-1.25, -1.5 ) => ambos juntos
