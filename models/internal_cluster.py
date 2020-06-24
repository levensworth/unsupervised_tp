import numpy as np
import math

class Cluster:
    # just an abstraction
    pass
# Necessary class for dendrogram
class AbstractNode(Cluster):
    def __init__(self, tag=None):
        self.sub_clusters = []
        self.tag = tag

    def get_data(self):
        return self.get_sub_clusters()

    def add_sub_cluster(self, cluster):
        self.sub_clusters.append(cluster)

    def get_sub_clusters(self):
        '''
        This is an idempotent function
        It always returns a copy of the original
        '''
        return self.sub_clusters.copy()

    def remove_sub_cluster(self, sub_cluster):
        self.sub_clusters.remove(sub_cluster)

    def get_tag(self):
        return self.tag

    def set_tag(self, tag):
        self.tag = tag

    def __lt__(self, other):
        # it always return as equals
        return 0

    def __eq__(self, value):
        if type(self) != type(value):
            return False
        return self.sub_clusters == value.get_sub_clusters()

    def get_max_distance(self, value):
        '''
        Assumes value and node value are comparable
        '''
        if not isinstance(value, TerminalNode):
            # it's not a terminal node
            return 0

        if len(self.sub_clusters) == 0:
            # stop flow
            return 0

        max_distance = 0
        for node in self.sub_clusters:

            # recursive call
            sub_distance = node.get_max_distance(value)
            max_distance = max(max_distance, sub_distance)

        return max_distance

    def get_min_distance(self, value):
        '''
        Assumes value and node value are comparable
        '''
        if not isinstance(value, TerminalNode):
            # it's not a terminal node
            return math.inf

        if len(self.sub_clusters) == 0:
            # stop flow
            return math.inf

        min_distance = math.inf
        for node in self.sub_clusters:

            # recursive call
            sub_distance = node.get_min_distance(value)
            min_distance = min(min_distance, sub_distance)

        return min_distance

    def get_all_distance(self, value):
        '''
        Assumes value and node value are comparable.
        Also, it just returns the vector with all distances
        '''
        distances = []
        for node in self.sub_clusters:
            distances += node.get_all_distance(value)

        return distances

    def get_centroid(self):
        node_sum = self._get_sum()
        count = self._get_node_count()
        return node_sum / float(count)

    def _get_sum(self):
        total = 0
        for node in self.sub_clusters:
            total += node._get_sum()
        return total

    def _get_node_count(self):
        '''
        Only consider leaf nodes
        '''
        total = 0
        for node in self.sub_clusters:
            total = node._get_node_count()

        return total

    def get_list_of_nodes(self):
        '''
        Return a list of all subnodes and given distances
        '''
        sub_nodes = []
        for node in self.sub_clusters:
            sub_nodes += node.get_list_of_nodes()
        return sub_nodes

class TerminalNode(Cluster):
    '''
    Represents a leaf
    '''
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_sub_clusters(self):
        return [self]

    def get_centroid(self):
        return self.data

    def get_list_of_nodes(self):
        return [self.data]

    def get_max_distance(self, value):

        if isinstance(value, AbstractNode):
            return value.get_max_distance(self)

        if type(value) != type(self):
            raise AttributeError
        # else we can compare the two of them
        return np.linalg.norm(value.get_data() - self.data)

    def get_min_distance(self, value):

        if isinstance(value, AbstractNode):
            return value.get_min_distance(self)

        if type(value) != type(self):
            raise AttributeError
        # else we can compare the two of them
        return np.linalg.norm(value.get_data() - self.data)

    def get_all_distance(self, value):
        if isinstance(value, AbstractNode):
            return value.get_all_distance(self)

        if type(value) != type(self):
            raise AttributeError
        return [np.linalg.norm(value.get_data() - self.data)]

    def _get_sum(self):
        return self.data

    def _get_node_count(self):
        return 1

    def __str__(self):
        return str(self.data)

    def __lt__(self, other):
        return 0
