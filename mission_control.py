import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# for mesh grid view
from matplotlib.collections import LineCollection
from mpl_toolkits import mplot3d
# for confusion matrix printing
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from models.kmeans import KMeans

from models.kohonen import Kohonen

from models.hierarchy_tree import HierarchyTree

# configure logging library for timestamp format
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def load_data(path, separator=',', encoding='utf-8'):
    df = pd.read_csv(path, sep= separator, encoding=encoding)
    return df

def complete_data(dataframe):
    '''
    Completes the missing entries based on a prevoius data analysis
    Params:
    - dataframe: pandas dataframe containing data to change

    outputs:
    - dataframe
    '''
    df = dataframe.sort_values(by=['age'])
    df.fillna(inplace=True, method='ffill')
    return df


def normalize_data(df):
    '''
    Normalize dataframe values base on col level
    '''
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.Normalizer()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def plot_grid(x, y, z, ax=None, **kwargs):
    '''
    Plot mesh grid for kohonen network results
    Params:
    - x: numpy
    '''
    ax = ax or plt.gca()
    # segs1 = np.stack((x,y), axis=2)
    # segs2 = segs1.transpose(1,0,2)
    # ax.add_collection(LineCollection(segs1, **kwargs))
    # ax.add_collection(LineCollection(segs2, **kwargs))
    # ax.autoscale()
    ax.plot_surface(x, y, z, **kwargs)


def predict_logistic_regression(train_data, test_data, train_lables, test_labels):
    '''
    predict if the value of sigdz column base on numerical values and prints a
    cofnussion matrix
    Params:
    - train_data: numpy array with scaled input
    - test_data: same as train .. but should be include in the aformationed set
    - train_labels: array like object with desired output for train data
    - test_labels: idem train_labels for test_data

    '''

    # train logistic regression model
    # model = sm.Logit(exog=train_data, endog=train_labels)
    model = LogisticRegression(n_jobs=3, C=0.3)
    model.fit(train_data, train_labels)
    # now we test results
    predictions = model.predict(test_data)
    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    # print a nice matrix
    plot_confusion_matrix(cm, target_names=['Negative', 'Positive'], title='Logistic Regression')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

if __name__ == '__main__':
    df = load_data('data/acath.csv', separator=';')
    # data = df[['age', 'cad.dur', 'choleste']].to_numpy()
    df = complete_data(df)
    data = df[['age', 'cad.dur', 'choleste']]
    label = df['sigdz'].to_list()

    # normalize data

    data = normalize_data(data)
    data = data.to_numpy()
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=0.95)

    # exercise b
    # predict_logistic_regression(train_data, test_data, train_labels, test_labels)

    # exercise d
    # # we add sex to variable
    # data = df[['sex', 'age', 'cad.dur', 'choleste']]
    # data = normalize_data(data)
    # data = data.to_numpy()
    # train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=0.8)

    # predict_logistic_regression(train_data, test_data, train_labels, test_labels)

    # ===== show 3d entry data =======
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], 'bo')
    # plt.show()
    # ================================

    # ====================
        # KOHONEN
    '''
    rows = 4
    cols = 4
    net = Kohonen(logging, (rows, cols, 3),
                learning_method='linear',
                radius_method='linear',
                learning_rate=0.3,
                max_epochs=200)
    net.fit(input_data=train_data)
    output_net = net.predict(test_data)
    '''
    # map output to classes and analyze each one
    # for row in range(rows):
    #     for col in range(cols):
    #         # analize individual node
    #         cluster_members = {}
    #         cluster_size = 0
    #         for idx in range(len(output_net)):
    #             if output_net[idx] == (row, col):
    #                 cluster_members[test_labels[idx]] = cluster_members.get(test_labels[idx], 0) + 1
    #                 cluster_size += 1

    #              # make piechart for each cluster
    #         distribution = []
    #         pie_labels = []
    #         for key, value in cluster_members.items():
    #             distribution.append(value / cluster_size)
    #             pie_labels.append(key)
    #         if cluster_size != 0 :
    #             plt.pie(distribution, labels=pie_labels, autopct='%1.1f%%',
    #             shadow=True, startangle=90)
    #             plt.xlabel('Cluster composition (cluster size {})'.format(cluster_size))
    #             plt.show()

    # === how to plot mesh ====
    '''
    grid_x = np.zeros((rows, cols))
    grid_y = np.zeros((rows, cols))
    grid_z = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            grid_x[row, col] = net.net[row, col, 0]
            grid_y[row, col] = net.net[row, col, 1]
            grid_z[row, col] = net.net[row, col, 2]

    fig, ax = plt.subplots()
    ax = plt.axes(projection="3d")
    plot_grid(grid_x, grid_y, grid_z, ax=ax, color="C0")
    plt.show()
    '''

    # ==== end plot ======

    # ====================

    # ====================
        # HIERARCHY
    # TODO: test in a less intensive dataset
    model = HierarchyTree(logging, 'centroid')
    # this model doesn't need fitting as it constructs a new dendrogram for any given input
    dendogram = model.predict(test_data)
    # i only need 1 level of depth as
    sub_clusters = model.clusterize(dendogram, 1.0)

    for cluster in sub_clusters:
        # now we print each cluster composition
        cluster_size = len(cluster)
        pred = {}
        for node in cluster:
            # find test index
            index = np.where(np.all(test_data==node,axis=1))[0]
            label = test_labels[int(index)]
            pred[label] = pred.get(label, 0) + (1.0/cluster_size)

        distribution = []
        pie_labels = []
        for key, value in pred.items():
            distribution.append(value)
            pie_labels.append(key)

        plt.pie(distribution, labels=pie_labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        plt.xlabel('Cluster composition (cluster size {})'.format(cluster_size))
        plt.show()

    # ====================


    # ===================
        # KMEANS

    n_clusteers = 2
    model = KMeans(n_clusteers, logging, max_iter=300)

    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)

    # we analyze the composition of each cluster
    possible_labels = [0,1]
    for cluster in range(n_clusteers):
        cluster_members = {}
        cluster_size = 0
        for idx in range(len(predictions)):
            # check if it's cluster member
            if predictions[idx] == cluster:
                # add real value
                cluster_members[test_labels[idx]] = cluster_members.get(test_labels[idx], 0) + 1
                cluster_size += 1
        # make piechart for each cluster
        distribution = []
        pie_labels = []
        for key, value in cluster_members.items():
            distribution.append(value / cluster_size)
            pie_labels.append(key)

        plt.pie(distribution, labels=pie_labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        plt.xlabel('Cluster composition (cluster size {})'.format(cluster_size))
        plt.show()



    # plt.scatter(test_data)
    # ===================
