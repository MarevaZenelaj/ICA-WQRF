# Based on https://github.com/tkipf/ica/blob/master/ica/aggregators.py
import numpy as np
import scipy.sparse as sp
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    md = __import__(module)
    for comp in parts[1:]:
        md = getattr(md, comp)
    return md


class Classifier():
    def __init__(self, scikit_classifier_name, params):
        classifer_class = get_class(scikit_classifier_name)
        self.clf = classifer_class(**params)

    def fit(self, graph, train_indices):
        raise NotImplementedError

    def predict(self, graph, test_indices, conditional_node_to_label_map=None):
        raise NotImplementedError


class LocalClassifier(Classifier):

    def fit(self, graph, tr_ids):
        features = []
        labels = []

        for node in tr_ids:
            feat_node = graph.nodes[node]['features']
            label_node = graph.nodes[node]['label']

            features.append(feat_node)
            labels.append(label_node)

        features = pd.DataFrame(features)

        self.clf.fit(features, labels)
        return
    
    
    def predict(self, graph, test_ids, conditional_node_to_label_map=None):
        features_test = []
        labels_test = []

        for node in test_ids:
            feat_node = graph.nodes[node]['features']
            features_test.append(feat_node)

        features_test = pd.DataFrame(features_test)

        predicted_labels = self.clf.predict(features_test)
        
        return predicted_labels


class RelationalClassifier(Classifier):
    def __init__(self, scikit_classifier_name, aggregator, params):
        super(RelationalClassifier, self).__init__(scikit_classifier_name, params)
        self.aggregator = aggregator
        
        
    def fit(self, graph, train_ids, local_classifier, bootstrap):
        conditional_map = {}

        if bootstrap:
            predictclf = local_classifier.predict(graph, train_ids)
            conditional_map = self.conditional_mapping(conditional_map, predictclf, train_ids)
        
        for node in train_ids:
            conditional_map[node] = graph.nodes[node]['label']
            
        features = []
        aggregates = []
        labels = []
        
        for node in train_ids:
            features.append(graph.nodes[node]['features'])
            labels.append(graph.nodes[node]['label'])
            ag_ = self.aggregator.aggregate(graph, node, conditional_map)
            aggregates.append(ag_)
            
        
        aggregates = pd.DataFrame(aggregates)
        features = pd.DataFrame(features)
        
        aggregates.index = features.index
        
        features = pd.concat([features, aggregates], axis=1)
        
        self.clf.fit(features, labels)    


    def predict(self, graph, test_ids, conditional_map=None):
        features = []
        aggregates = []
        
        for node in test_ids:
            features.append(graph.nodes[node]['features'])
            ag_ = self.aggregator.aggregate(graph, node, conditional_map)
            aggregates.append(ag_)

        aggregates = pd.DataFrame(aggregates)
        features = pd.DataFrame(features)
        
        aggregates.index = features.index
        
        features = pd.concat([features, aggregates], axis=1)

        return self.clf.predict(features)        
        
        
    def conditional_mapping(self, conditional_map, predictions, ids):
        for id_ in range(len(ids)):
            conditional_map[ids[id_]] = predictions[id_]
            
        return conditional_map
        


class ICA(Classifier):
    def __init__(self, local_classifier, relational_classifier, bootstrap, max_iteration=20):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.bootstrap = bootstrap
        self.max_iteration = max_iteration

    def fit(self, graph, train_indices):
        self.local_classifier.fit(graph, train_indices)
        self.relational_classifier.fit(graph, train_indices, self.local_classifier, self.bootstrap)

    def predict(self, graph, eval_indices, test_indices, conditional_node_to_label_map=None):

        # the node's labelling is done all at once, it should go iteratively one by one until all the labels are stabilized
        predictclf = self.local_classifier.predict(graph, eval_indices)
        conditional_node_to_label_map = self.conditional_mapping(conditional_node_to_label_map,
                                                         predictclf, eval_indices)

        relation_predict = []

        temp = []

        for iter_ in range(self.max_iteration):

            for x in eval_indices:                
                temp.append(x)
                rltn_pred = list(self.relational_classifier.predict(graph, temp, conditional_node_to_label_map))
                conditional_node_to_label_map = self.conditional_mapping(conditional_node_to_label_map, rltn_pred, temp)
                temp.remove(x)
                
        for node in test_indices:
            relation_predict.append(conditional_node_to_label_map[node])
            
        return relation_predict

    def conditional_mapping(self, conditional_map, predictions, ids):
        for id_ in range(len(ids)):
            conditional_map[ids[id_]] = predictions[id_]
            
        return conditional_map



class CombinedRF:

    def __init__(self, K, data, labels, dtc):
        self.decision_tree_classifier = dtc
        self.K = K
        
        self.train = None
        self.val = None
        self.y_train = None
        self.y_val = None
        
        self.data = data
        self.labels = labels
        
        self.subsamples = None
        
        self.weights = []
        self.subclassifiers = []
        
    
    def split_data(self):
        
        x_train, x_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.25)

        self.train = pd.DataFrame(x_train)
        self.y_train = y_train
        self.val = pd.DataFrame(x_val) 
        self.y_val = y_val


    
    def bootstrap(self):
        
        self.split_data()
        subsamples = []
        for i in range(self.K):
            subsamples.append(resample(self.train.index))
        self.subsamples = subsamples
    
    
    def f_measure(self, class_name, preds, reals):
        f_measure = pd.DataFrame(classification_report(reals, preds, output_dict=True)).loc['f1-score', class_name]
        return f_measure 
    
    
    def fit(self, new_model=None):
        self.subclassifiers = []
        self.weights = []
        
        self.bootstrap()
        for subsample in self.subsamples:
            
            data = self.train.loc[subsample]
            
            labels = self.y_train[subsample]
            
            sample_weights = []
            for el in labels:
                if el == 0:
                    sample_weights.append(2)
                if el == 1:
                    sample_weights.append(1)

            if new_model is None:
                model = self.decision_tree_classifier.fit(data, labels, sample_weight=sample_weights)
            else:
                model = new_model.fit(data, labels, sample_weight=sample_weights)
            
            weight = self.f_measure('0', model.predict(self.val), self.y_val)
            self.subclassifiers.append(model)
            self.weights.append(weight)

            
    
    def predict_one_at_a_time(self, sample, weights=True):
        predicts = []
        for classifier in self.subclassifiers:
            
            predicts.append(classifier.predict(sample))
        
        results = pd.DataFrame(self.weights, columns=['weight'])
        results = pd.concat([results, pd.Series(predicts, name='prediction')],axis=1)


        class_0_prob = results[results['prediction']==0]['weight'].sum()
        class_1_prob = results[results['prediction']==1]['weight'].sum()
        
        if weights is False:
            return results['prediction'].value_counts().idxmax()
        else:
            return 0 if class_0_prob > class_1_prob else 1
    
    def predict(self, test, weights=True):
        predicts = []
        for i in range(test.shape[0]):
            sample = test.iloc[i]
            sample = sample.to_numpy()
            sample = sample.reshape([1,-1])
            predicts.append(self.predict_one_at_a_time(sample, weights))        
        
        return predicts