from scipy.sparse.csgraph import dijkstra
import numpy as np
import networkx as nx
from pathlib import Path

import pandas as pd


# read in source and target sets
source_file = Path("./source.txt")
target_file = Path("./target.txt")

G = nx.Graph()
# G is the graph and P is a subgraph of: S intersects T
P = nx.Graph()
# Example with 10 nodes and 15 edges
# In multi_adj: The first label in a line is the source node label followed by the node degree d.
# The next d lines are target node labels and optional edge data.


# 1. Initialize a pathway P with all nodes and flag them as unvisited, having weights different from one another so there's no ties
nodes = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'k', 'h', 'i')
elist = [('a', 'b', 1.0), ('a', 'c', 0.4), ('b', 'd', 0.5), ('c', 'e', 0.7),
         ('b', 'e', 0.8), ('b', 'f', 0.5), ('c', 'k',
                                            0.3), ('d', 'e', 0.5), ('d', 'h', 0.2),
         ('d', 'i', 1.0), ('f', 'e', 0.6), ('g', 'e', 0.1), ('e', 'h', 0.5), ('k', 'h', 0.7), ('k', 'i', 1.0)]

source_set = (['a', 'b', 'c'])
target_set = (['e', 'f', 'g'])
G.add_weighted_edges_from(elist)  # add these and flag as unvisited

# calculate the total number of sources x targets
paths = len(source_set) * len(target_set)


# determine nodes on path between sources and targets
print(list(G.neighbors('a')))

# calculate score of intermediate nodes:


# calculate score for each node in the list, EXCLUDE the sources and targets
def calculate_score(node):
    source_connected = 0
    target_connected = 0
    if node not in source_set and node not in target_set:
        for source in source_set:
            if source in list(G.neighbors(node)):
                source_connected += 1
        for target in target_set:
            if target in list(G.neighbors(node)):
                target_connected += 1
    return (source_connected*target_connected)/(len(source_set) * len(target_set))


calculate_score('d')

'''
adjacancy_matrix = []

for source in source_set:
    temp = []
    for target in target_set:
        temp.append(nx.dijkstra_path_length(G, source, target))
    adjacancy_matrix.append(temp)

print(adjacancy_matrix)
'''
# use Dijkstra’s algorithm to find the shortest weighted path between a and f (and repeat for all source and target nodes as well)
# print(nx.dijkstra_path(G, {'a', 'b', 'c'}, {'e', 'f', 'g'}))

# print(nx.dijkstra_path_length(G, 'a', 'f'))
# print(nx.dijkstra_path(G, 'a', 'f'))

# nx.draw(G)

'''
# then add these nodes to a set of visited nodes
#for e in list(G.edges):

    # cannot iterate thru all source and target nodes yet
   # for node in source_set:
        #for target in target_set:
            #if e in nx.dijkstra_path(G, node, target):
                #P.add_edge(e)
print(P)
'''

# maybe a better approach would be:
# for e in list of edges
# if e contains a node from the source set and a node from the target set
# run dijkstra_path on (G, source, target)


def btb_implement(S, T, G):
    # Step 1: Initialize the pathway P with nodes in S ∩ T
    P = nx.Graph()
    common_nodes = set(S) & set(T)
    P.add_nodes_from(common_nodes)
    visited = set()

    while True:
        # Step 2: Calculate the distance matrix D|S|×|T| between nodes in S and T
        subgraph_S = G.subgraph(S)
        subgraph_T = G.subgraph(T)
        subgraph_nodes = list(subgraph_S.nodes) + \
            list(subgraph_T.nodes)  # all nodes in S+T
        node_indices = {node: i for i, node in enumerate(subgraph_nodes)}
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=subgraph_nodes)
        distances = dijkstra(adjacency_matrix, directed=False, indices=[
                             node_indices[node] for node in common_nodes])

        # Step 3: Select the shortest path in D that connects a 'not visited' and a 'visited' node in P
        # or a 'not visited' node in S to a 'not visited' node in T
        path_exists = False
        for i, source_node in enumerate(subgraph_S.nodes):
            if source_node not in visited:
                for j, target_node in enumerate(subgraph_T.nodes):
                    if target_node not in visited:
                        if distances[i, j] != np.inf:
                            path = nx.shortest_path(
                                G, source=source_node, target=target_node, weight='weight')
                            path_exists = True
                            break
                if path_exists:
                    break

        if not path_exists:
            for i, source_node in enumerate(subgraph_S.nodes):
                if source_node not in visited:
                    for j, target_node in enumerate(subgraph_S.nodes):
                        if target_node != source_node:
                            if distances[i, j] != np.inf:
                                path = nx.shortest_path(
                                    G, source=source_node, target=target_node, weight='weight')
                                path_exists = True
                                break
                if path_exists:
                    break

        if not path_exists:
            break

        # Step 4: Add nodes and edges of the selected path to P and mark nodes as 'visited'
        P.add_nodes_from(path)
        P.add_edges_from(nx.utils.pairwise(path))
        visited.update(path)

        # Step 5: Update D to include all distances to the nodes in P
        subgraph_P = G.subgraph(P.nodes)
        subgraph_nodes = list(subgraph_P.nodes)
        node_indices = {node: i for i, node in enumerate(subgraph_nodes)}
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=subgraph_nodes)
        distances = dijkstra(adjacency_matrix, directed=False, indices=[
                             node_indices[node] for node in subgraph_nodes])

    # Step 7: Export final pathway P
    return P
