# @Author: Mareva Zenelaj
# Date: 22 November 2019

# Edited version of https://github.com/tkipf/ica/blob/master/ica/graph.py, but it is implemented with networkx and pandas instead

import networkx as nx
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

class Graph(object):

    def __init__(self):
        # nodes and edges are useless attributes here, but might be useful later, hence they stay :D
        self.nodes = []
        self.edges = []
        self.Graph = nx.Graph()


    def give_graph(self, graph):
    	self.Graph = graph


    def add_nodes(self, nodes):
        self.nodes = nodes
        self.Graph.add_nodes_from(self.nodes)


    def add_edges(self, edges):
        self.edges = edges
        self.Graph.add_edges_from(self.edges)


    def update_nodes(self, features, labels):
        # features should be a pandas dataframe
        if features is not None:     
            if type(features) != type(pd.DataFrame()):
                features = pd.DataFrame(features)

        if type(labels) != type(pd.DataFrame()) and type(labels) != type(pd.Series()):
            labels = pd.Series(labels)

        # ids = features.index
        # ids should be the same as nodes in G
        # check the consistency

        for node in self.Graph.nodes():
            if features is not None:
                self.Graph.nodes[node]['features'] = features.loc[node]
            self.Graph.nodes[node]['label'] = labels.loc[node]


    def add_weights(self, weights):
        # might not be necessary for the simple ICA but we might want to check the strength of the neighborhood. 

        if type(weights) != type(dict()):
            error = 'Weights are inputted only by a dict object of the form {tuple(node_1, node_2): weight}'
            print(error)
        else:
            for edge in self.Graph.edges():
                self.Graph.edges[edge]['weights'] = weights[edge]


    def get_neighbors(self, node):

        return self.Graph[node]


    def display_neighborhood(self, node):

        neighbors = list(dict(self.Graph[node]).keys())
        neighbors.append(node)

        subgraph = self.Graph.subgraph(neighbors)

        pos = nx.spring_layout(subgraph)

        labels = [[node, subgraph.nodes[node]['label']] for node in subgraph.nodes]

        dict_labels = dict(labels)

        edges = []
        edges_x = []
        edges_y = []
        
        for edge in subgraph.edges():
            edge_1, edge_2 = edge
            pos_1 = pos[edge_1]
            pos_2 = pos[edge_2]
            edges_x.append(pos_1[0])
            edges_x.append(pos_2[0])
            edges_x.append(None)
            edges_y.append(pos_1[1])
            edges_y.append(pos_2[1])
            edges_y.append(None)
            
        nodes_x = []
        nodes_y = []
        
        for node in subgraph.nodes():
            nodes_x.append(pos[node][0])
            nodes_y.append(pos[node][1])
            
        node_trace = go.Scatter(x=nodes_x, y=nodes_y,mode='markers',
                                marker=dict(color=[],size=10,line_width=2))
        
        edge_trace = go.Scatter(x=edges_x, y=edges_y,line=dict(width=1, color='#888'),
                                hoverinfo='none',mode='lines')
        
        
        
        if 'features' in subgraph.nodes[list(subgraph.nodes)[0]].keys():            
            node_colors = []
            hover_templates = []
            for node in subgraph.nodes:
                if node in dict_labels.keys():
                    node_color = 'blue' if dict_labels[node] == 0 else 'green'
                    hov_ = "Employee Id: {}<br>".format(node)

                    features = subgraph.nodes[node]['features']

                    for ind in features.index:
                        hov_ += "{}: {}<br>".format(ind, features[ind])
                    hov_ += "<extra></extra>"
                else:
                    node_color = 'grey'

                node_colors.append(node_color)
                hover_templates.append(hov_)

            node_trace.marker.color = node_colors
            node_trace.hovertemplate = hover_templates
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                    titlefont_size=16, showlegend=False, margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

       
        return fig


