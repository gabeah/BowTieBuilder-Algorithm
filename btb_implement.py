import networkx as nx
from pathlib import Path

import pandas as pd


# read in source and target sets
source_file = Path("./source.txt")
target_file = Path("./target.txt")

G = nx.Graph()
P = nx.Graph()
# Example with 10 nodes and 15 edges
# In multi_adj: The first label in a line is the source node label followed by the node degree d.
# The next d lines are target node labels and optional edge data.

# 1. Initialize a pathway P with all nodes and flag them as unvisited
elist = [('a', 'b', 1.0), ('a', 'c', 1.0), ('b', 'd', 1.0), ('c', 'e', 1.0),
         ('b', 'e', 1.0), ('b', 'f', 1.0), ('c', 'k',
                                            1.0), ('d', 'e', 1.0), ('d', 'h', 1.0),
         ('d', 'i', 1.0), ('f', 'e', 1.0), ('g', 'e', 1.0), ('e', 'h', 1.0), ('k', 'h', 1.0), ('k', 'i', 1.0)]
source_set = (['a', 'b', 'c'])
target_set = (['e', 'f', 'g'])
G.add_weighted_edges_from(elist)

# use Dijkstra’s algorithm to find the shortest weighted path between a and f (and repeat for all source and target nodes as well)
# print(nx.dijkstra_path(G, {'a', 'b', 'c'}, {'e', 'f', 'g'}))


print(nx.dijkstra_path(G, 'a', 'f'))

nx.draw(G)

'''
# then add these nodes to a set of visited nodes
#for e in list(G.edges):

    # cannot iterate thru all source and target nodes yet
   # for node in source_set:
        for target in target_set:
            if e in nx.dijkstra_path(G, node, target):
                P.add_edge(e)
print(P)
'''

# maybe a better approach would be:
# for e in list of edges
# if e contains a node from the source set and a node from the target set
# run dijkstra_path on (G, source, target)

# approach that makes more sense:
