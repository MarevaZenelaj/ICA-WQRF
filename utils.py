"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import os.path
import graph as Graph
from aggregators import Count
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def pick_aggregator(agg, domain_labels):
    if agg == 'count':
        aggregator = Count(domain_labels)
    elif agg == 'prop':
        aggregator = Prop(domain_labels)
    else:
        raise ValueError('Invalid argument for agg (aggregation operator): ' + str(agg))
    return aggregator


def create_map(graph, train_indices):
    conditional_map = {}
    for node in train_indices:
        conditional_map[node] = graph.nodes[node]['label']
    return conditional_map


def load_data():
    data = pd.read_csv('data/real-data-28Nov.csv')
    data.index = data['uid']
    data = data.drop(['uid', 'uid.1'], axis=1)

    labels = data['E_target'].to_numpy()
    data = data.drop('E_target', axis=1)
    
    feats = data

    adjacency_matrix = pd.read_csv('data/adj_matrix_28Nov.csv', index_col=0)
    # adjacency_matrix.index = adjacency_matrix.uid
    # adjacency_matrix = adjacency_matrix.drop('uid', axis=1)
    
    edges = []

    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[0]):
            if adjacency_matrix.iloc[i,j] > 0:
                id_1, id_2 = adjacency_matrix.columns[i], adjacency_matrix.columns[j]
                edges.append((id_1,id_2))    

    
    index_ = list(feats.index)
    random.shuffle(index_)
    feats = feats.loc[index_]
    labels = pd.Series(labels, index=feats.index)
    graph = Graph.Graph()     
    graph.add_nodes(list(adjacency_matrix.index))
    graph.add_edges(edges)
    graph.update_nodes(feats, labels)

    ids = list(feats.index)
    random.shuffle(ids)
    train_, test_ = train_test_split(ids, test_size=0.2)
    train_, val_ = train_test_split(train_, test_size=0.2)

    #train_, val_, test_ = ids[:int(len(ids)*40/100)], ids[int(len(ids)*40/100):int(len(ids)*80/100)], ids[int(len(ids)*80/100):len(ids)]
    
    val_ = val_ + test_

    domain_labels = list(labels.unique())

    return feats, labels, train_, val_, test_, graph, domain_labels