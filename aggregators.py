# Credits to https://github.com/tkipf/ica/blob/master/ica/aggregators.py
# Edited to fit with graph.py
import numpy as np
import scipy.sparse as sp

class Aggregator(object):
    def __init__(self, domain_labels):
        self.domain_labels = domain_labels  # The list of labels in the domain

    def aggregate(self, graph, node, conditional_node_to_label_map):
        raise NotImplementedError


class Count(Aggregator):
    def aggregate(self, graph, node, conditional_node_to_label_map):
        neighbor_undirected = []

        for x in self.domain_labels:
            neighbor_undirected.append(0.0)
            
        for n_node in list(dict(graph[node]).keys()):
            if n_node in conditional_node_to_label_map.keys():
                index = self.domain_labels.index(conditional_node_to_label_map[n_node])
                neighbor_undirected[index] += 1.0
        
        return neighbor_undirected


class Prop(Aggregator):
    def aggregate(self, graph, node, conditional_node_to_label_map):
        cntag = Count(self.domain_labels)
        cnt_agg = cntag.aggregate(graph, node, conditional_node_to_label_map)
        total_sum = sum(cnt_agg)
        if total_sum > 0:
            for r in range(len(cnt_agg)):
                cnt_agg[r] /= total_sum
        p_list = cnt_agg
        return p_list
