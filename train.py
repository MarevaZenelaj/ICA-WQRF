# The ica implementation is an edited version of https://github.com/tkipf/ica

from utils import load_data, pick_aggregator, create_map
from classifiers import LocalClassifier, RelationalClassifier, ICA, CombinedRF
from scipy.stats import sem
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
import random
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data/real-data-28Nov.csv')

labels = data['E_target'].to_numpy()
data = data.drop(['E_target', 'uid.1'], axis=1)

adjacency_matrix = pd.read_csv('data/adj_matrix_28Nov.csv')
adjacency_matrix.index = adjacency_matrix.uid
adjacency_matrix = adjacency_matrix.drop('uid', axis=1)

adjacency_matrix_weights = pd.read_csv('data/adjacency_matrix_no_weights.csv', index_col=0)
adjacency_matrix_weights = adjacency_matrix_weights.loc[adjacency_matrix.index.to_list()]
adjacency_matrix_weights = adjacency_matrix_weights[adjacency_matrix.index.to_list()]

adjacency_matrix_similarity = pd.read_csv('data/adj_similarity_matrix.csv', index_col=0)
adjacency_matrix_similarity = adjacency_matrix_similarity.loc[adjacency_matrix.index.to_list()]
adjacency_matrix_similarity = adjacency_matrix_similarity[adjacency_matrix.index.to_list()]

data = MinMaxScaler().fit_transform(data)

def run_ICA(classifier_name, classifier_args, num_iter):

    features, labels, train_, val_, test_, graph, domain_labels = load_data()

    # run training
    t_begin = time.time()

    # random ordering
    np.random.shuffle(val_)

    y_true = [graph.Graph.nodes[node]['label'] for node in test_]

    local_clf = LocalClassifier(classifier_name, classifier_args)

    agg = pick_aggregator('count', domain_labels)

    relational_clf = RelationalClassifier(classifier_name, agg, classifier_args)

    ica = ICA(local_clf, relational_clf, True, max_iteration=num_iter)

    ica.fit(graph.Graph, train_)

    print('Model fitting done...')

    conditional_node_to_label_map = create_map(graph.Graph, train_)

    ica_predict = ica.predict(graph.Graph, val_, test_, conditional_node_to_label_map)

    print('Model prediction done...')

    ica_accuracy = accuracy_score(y_true, ica_predict)
    t_end = time.time()
    print(classification_report(y_true, ica_predict))

    print(ica_accuracy)
    elapsed_time = t_end - t_begin
    print('Start time: \t\t' + time.strftime("%H:%M:%S", time.gmtime(t_begin)))
    print('Elapsed time: \t\t' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print('End time: \t\t' + time.strftime("%H:%M:%S", time.gmtime(t_end)))



def run_NMF():
    
    true_labels = labels

    adjs = [adjacency_matrix, adjacency_matrix_weights, adjacency_matrix_similarity]
    names = ['Adjacency Matrix: no weights\n', 'Adjacency Matrix: likes-dislikes weights\n', 'Adjacency Matrix: similarity weights\n']

    for adj, name in zip(adjs, names):

        nmf_factorization = non_negative_factorization(adj, n_components=2, init='random')
        W = nmf_factorization[0]
        W = pd.DataFrame(W)

        H = nmf_factorization[1]
        H = pd.DataFrame(H)

        clusters = [1 if (H.iloc[0,i] < H.iloc[1,i]) else 0 for i in range(H.shape[1])]
        clusters = pd.Series(clusters)
        predicted_labels = list(clusters)

        print(name)
        print(classification_report(true_labels, predicted_labels))
        print('--------------------------------------------\n')


def run_SC():

    y_true = labels

    sc = SpectralClustering(n_clusters=2, affinity="precomputed", n_init=200)

    labels_ = sc.fit_predict(adjacency_matrix)
    print('Adjacency Matrix: no weights\n')
    print(classification_report(y_true, labels_))
    print('----------------------------------------------------\n')

    labels_ = sc.fit_predict(adjacency_matrix_weights)
    print('Adjacency Matrix: likes-dislikes weights\n')
    print(classification_report(y_true, labels_))
    print('----------------------------------------------------\n')

    labels_ = sc.fit_predict(adjacency_matrix_similarity)
    print('Adjacency Matrix: similarity weights\n')
    print(classification_report(y_true, labels_))
    print('----------------------------------------------------\n')


def run_WQRF(n_estimators):
    
    data_ = data.copy()

    feat_imp = pd.read_csv('data/feat_importance_28_Nov.csv', index_col=0)
    feat_imp = pd.Series(feat_imp['0'], index=feat_imp.index)
    feat_imp = feat_imp.sort_values(ascending=False)
    feat_imp = feat_imp[feat_imp > 0.01]
    data_ = data_[feat_imp.index]
    scaler = MinMaxScaler()
    data_ = scaler.fit_transform(data_)
    data_ = pd.DataFrame(data_)
    labels_ = pd.Series(labels_)

    X_train, X_test, y_train, y_test = train_test_split(data_, labels_, test_size=0.25)

    dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       splitter='best')

    crf = CombinedRF(n_estimators, X_train, y_train, dtc)
    crf.fit()
    preds = crf.predict(X_test, weights=True)
    print(classification_report(y_test, preds))
