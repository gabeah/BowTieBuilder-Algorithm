from scipy.sparse.csgraph import dijkstra
import numpy as np
import networkx as nx
from pathlib import Path
import sys
import pandas as pd

# read in source and target sets
'''
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


# determine nodes on path between sources and targets
print(list(G.neighbors('a')))

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

adjacancy_matrix = []

for source in source_set:
    temp = []
    for target in target_set:
        temp.append(nx.dijkstra_path_length(G, source, target))
    adjacancy_matrix.append(temp)

print(adjacancy_matrix)
# use Dijkstra’s algorithm to find the shortest weighted path between a and f (and repeat for all source and target nodes as well)
# print(nx.dijkstra_path(G, {'a', 'b', 'c'}, {'e', 'f', 'g'}))

# print(nx.dijkstra_path_length(G, 'a', 'f'))
# print(nx.dijkstra_path(G, 'a', 'f'))

# nx.draw(G)

# then add these nodes to a set of visited nodes
#for e in list(G.edges):

    # cannot iterate thru all source and target nodes yet
   # for node in source_set:
        #for target in target_set:
            #if e in nx.dijkstra_path(G, node, target):
                #P.add_edge(e)
print(P)

'''
# approach that makes more sense:

# Create a sample graph
P = nx.Graph()
G = nx.Graph()

G.add_edge('a', 'd', weight=0.5)
G.add_edge('a', 'c', weight=0.8)
G.add_edge('b', 'c', weight=1.2)
G.add_edge('b', 'd', weight=1.0)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('d', 'e', weight=1.5)
G.add_edge('e', 'f', weight=0.9)
G.add_edge('a', 'h', weight=0.2)
G.add_edge('h', 'd', weight=0.1)

# P.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 0.8), (2, 3, 1.2), (2, 4, 1.0), (3, 5, 0.7), (4, 5, 1.5), (5, 6, 0.9)])

# Define the sets of nodes S and T, containing strings
S = ['a', 'b', 'c']
T = ['d', 'e', 'f']

# 1. Initialize a pathway P with all nodes S ∩ T and flag them as unvisited (here: the edges are not added yet)
# P_nodes = []
common_nodes = set(S) & set(T)
P.add_nodes_from(common_nodes)
unvisited = P.nodes
print(P)
print(P.nodes)


# Step 2: Calculate the distance matrix D|S|×|T| between nodes in S and T

'''
D_path = []
for i in S:
    D_path.append([0]*len(T))
for i, s in enumerate(S):  # in source set
    for j, t in enumerate(T):  # in target set
        # return list of nodes in the shortest path
        D_path[i][j] = nx.dijkstra_path(G, s, t, weight='weight')
print(D_path)
visited = []
'''

D_path = {}
D = {}
for i in S:
    D_path[i] = {}
    D[i] = {}
    for j in T:
        D_path[i][j] = nx.dijkstra_path(G, i, j, weight='weight')
        D[i][j] = nx.dijkstra_path_length(G, i, j, weight='weight')

print(D_path)
visited = []
print(D)
# pass in the targets and sources to know the order, iterate thru the targets list

# 3. Find P^D(s,t) and add it to the pathway P
# 4. Add the nodes and edges of the selected path to P and flag all nodes in the pathway as 'visited'.

# Select the shortest path in D that connects a 'not visited' and a 'visited' node in P, or,
# if no such path exists, a 'not visited' node in S to a 'not visited' node in T.


def find_pd(D_path, visited, S, T, D, P):
    if visited == []:  # if no node has been visited
        node1 = 0
        node2 = 0
        for i in D.keys():
            for j in D[i].keys():
                if node1 == 0 or D[i][j] < D[node1][node2]:
                    node1 = i
                    node2 = j
        path = D_path[node1][node2]
        print(path)

        P.add_edges_from(zip(path, path[1:]), weight='weight')
        # add h to the set of visited nodes
        # visited.append(P.nodes)
        visited.extend([node1, node2])
        for entry in path:
            if entry not in visited:
                visited.append(entry)  # add all nodes in P to set of visited
# at this point: the set of visited nodes contains intermediate nodes

        print(f"Adding pathway between {node1} and {node2}")
        print(f"The pathway is {path}")
        print(f"The edges in P are {P.edges()}")
    else:
        # when at least one of the 2 nodes are visited
        lowest = float('inf')
        node1 = None
        node2 = None
        for i in D.keys():
            for j in D[i].keys():
                if D[i][j] < lowest:
                    if S[i] not in visited and T[j] in visited:
                        lowest = D[i][j]
                        node1 = i
                        node2 = j
                        visited.append(node1)
                    elif S[i] in visited and T[j] not in visited:
                        lowest = D[i][j]
                        node1 = i
                        node2 = j
                        visited.append(node2)

        path = D_path[node1][node2]
        P.add_edges_from(zip(path, path[1:]), weight='weight')
        for node in P.nodes:
            if node not in visited:
                visited.append(node)

        print(f"Adding pathway between {node1} and {node2}")
        print(f"The pathway is {path}")
        print(f"The edges in P are {P.edges}")

    print(f"The visited set is: {visited}")


find_pd(D_path, visited, S, T, D, P)
print(P.edges)
print(P.nodes)

# 5. Update D to include all distances to the nodes in P^D(s, t) (visited??). takes D, return renewed D
# the nodes in that path are flagged as 'visited', The method terminates when all nodes in D are flagged as 'visited'?? ,
# or, if for the remaining nodes, no path to any other node in S∩T exists.


# while set of visited is not set of all nodes in D, add 1 by 1 node in visited set to D
# add the distance of the new node to D, not the node

print(len(D))


def add_entry_to_D(G, entry, D, S, T):
    D[entry] = {}
    for i in S:
        D[i][entry] = nx.dijkstra_path_length(G, i, entry, weight='weight')
        for j in T:
            print(entry, j)
            print(D)
            D[entry][j] = nx.dijkstra_path_length(G, entry, j, weight='weight')
    return D


for node in visited:
    if node not in D.keys():
        add_entry_to_D(G, node, D, S, T)

# 6. Repeat steps 2–5 until every node in S is connected to some node in T, and vice versa if such a path exists in G.

'''

def wrapper():

    while set(S) - set(visited) or set(T) - set(visited):
        find_pd(D_path, visited, S, T, D, P)
        for node in visited:
            if D.get(node) is None:
                add_entry_to_D(G, node, D, S, T)

    print(P.edges)
    print(P.nodes)


def update_D(D):
    i = 0
    while i < len(D):
        j = 0
        while j < len(D[i]):
            for node in visited:
                if node not in D:
                    for i in S:
                        for j in T:
                            D[i+1][j] = nx.dijkstra_path_length(
                                G, node, t, weight='weight')
                            D[i][j+1]
                        return D


update_D(D)
print(D)



n = len(D)  # Current size (columns) of the distance matrix
m = len(D[0])  # Current rows of the distance matrix
# Calculate shortest paths from the new entry to existing entries
# shortest_paths = nx.single_source_dijkstra_path_length(G, entry, cutoff=None, weight='weight')
print(entry)
 # Create a new distance matrix with an additional row and column

new_D = []
new_D_path = []
for i in S:
        D.append([0]*len(T))
        D_path.append([0]*len(T))
        new_D = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        new_D_path = [[float('inf')] * (n + 1) for _ in range(n + 1)]
# Copy existing distances to the new distance matrix, and the new_D_path matrix
    for i in range(n):
        for j in range(m):
            new_D[i][j] = D[i][j]
            new_D_path[i][j] = D_path[i][j]

# Update the new distance matrix with paths (new_D_path) with shortest paths to existing entries

    for i in range(len(new_D)):
        for j in range(len(i)):
            # call nx.dij(node, t1), then nx.dij(s1, node)
            new_D_path[i][j] = nx.dijkstra_path_length(
                G, S[i], entry, weight='weight')
            new_D_path[len(new_D)][i] = nx.dijkstra_path_length(
                G, entry, T[i], weight='weight')

    # Update the distance from the new entry to itself as infinity
    new_D[len(new_D)][len(new_D)] = float('inf')
    return new_D


for node in visited:
    # if node not in D[]:
    # D_path
    if node
    add_entry_to_D(G, node, D)

print(S[1])

# a,b,c / d,e,f
D = [[1, 2, 3], [2, 3, 4]]
calculated_node = [a, b, c, d, e, f]

# g to D
calculated_node.append

sys.exit()



# when all sources and targets are in P (pathway), done with step 6.

# 7. Export final pathway P.

# look at dataframes in pandas - matrices manipulating
'''
