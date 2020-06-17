import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# for confusion matrix printing
import seaborn as sns
import matplotlib.pyplot as plt

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
    Normalize dataframe values base on a min max normalization
    '''
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

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
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=0.8)

    # exercise b
    predict_logistic_regression(train_data, test_data, train_labels, test_labels)

    # exercise d
    # we add sex to variable
    data = df[['sex', 'age', 'cad.dur', 'choleste']]
    data = normalize_data(data)
    data = data.to_numpy()
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=0.8)

    predict_logistic_regression(train_data, test_data, train_labels, test_labels)

    