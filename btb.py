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

# determine nodes on path between sources and targets
print(list(G.neighbors('a')))
'''
# pass in the targets and sources to know the order, iterate thru the targets list


def add_entry_to_D(G, entry, D, S, T, D_path):
    D[entry] = {}
    D_path[entry] = {}
    for i in S:
        D[i][entry] = nx.dijkstra_path_length(G, i, entry, weight='weight')
        D_path[i][entry] = nx.dijkstra_path(G, i, entry, weight='weight')
        for j in T:
            D[entry][j] = nx.dijkstra_path_length(G, entry, j, weight='weight')
            D_path[entry][j] = nx.dijkstra_path(G, entry, j, weight='weight')
    return D


def find_pd(D_path, visited, D, P):
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
                visited.append(entry)

        print(f"Adding pathway between {node1} and {node2}")
        print(f"The pathway is {path}")
        print(f"The edges in P are {P.edges()}")
        print(visited)
    else:
        # when at least one of the 2 nodes are visited
        lowest = float('inf')
        node1 = None
        node2 = None
        for i in D.keys():
            for j in D[i].keys():
                if D[i][j] < lowest:
                    if i not in visited and j in visited:
                        lowest = D[i][j]
                        node1 = i
                        node2 = j
                        visited.append(node1)
                    elif i in visited and j not in visited:
                        lowest = D[i][j]
                        node1 = i
                        node2 = j
                        visited.append(node2)
                        # print(visited)
        if node1 is None or node2 is None:  # if node1 and node2 are both unvisited
            node1 = 0
            node2 = 0
            for i in D.keys():
                for j in D[i].keys():
                    if node1 == 0 or D[i][j] < D[node1][node2]:
                        node1 = i
                        node2 = j
                        visited.extend([node1, node2])

        print(D_path)
        path = D_path[node1][node2]
        P.add_edges_from(zip(path, path[1:]), weight='weight')
        for node in P.nodes:
            if node not in visited:
                visited.append(node)

        print(f"Adding pathway between {node1} and {node2}")
        print(f"The pathway is {path}")
        print(f"The edges in P are {P.edges}")

    print(f"The visited set is: {visited}")
    return P

# while set of visited is not set of all nodes in D, add 1 by 1 node in visited set to D
# add the distance of the new node to D, not the node


# 6. Repeat steps 2–5 until every node in S is connected to some node in T, and vice versa if such a path exists in G.


# 4. Add the nodes and edges of the selected path to P and flag all nodes in the pathway as 'visited'.


# 5. Update D to include all distances to the nodes in P^D(s, t) (visited??). takes D, return renewed D
# the nodes in that path are flagged as 'visited', The method terminates when all nodes in D are flagged as 'visited'?? ,
# or, if for the remaining nodes, no path to any other node in S∩T exists.


def main():
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
    # Define the sets of nodes S and T, containing strings
    S = ['a', 'b', 'c']
    T = ['d', 'e', 'f']

    # 1. Initialize a pathway P with all nodes S ∩ T and flag them as unvisited (here: the edges are not added yet)
    # P_nodes = []
    common_nodes = set(S) & set(T)
    P.add_nodes_from(common_nodes)
    # unvisited = P.nodes
    visited = []
    # print(P)
    # print(P.nodes)

# Step 2: Calculate the distance matrix D|S|×|T| between nodes in S and T -> D (distance length) and D_path (all nodes on path)
    D_path = {}
    D = {}
    for i in S:
        D_path[i] = {}
        D[i] = {}
        for j in T:
            D_path[i][j] = nx.dijkstra_path(G, i, j, weight='weight')
            D[i][j] = nx.dijkstra_path_length(G, i, j, weight='weight')
    print(D_path)
    print(D)

# 3. Select the shortest path in D that connects a 'not visited' and a 'visited' node in P, or, if no such path exists, a 'not visited' node in S to a 'not visited' node in T.
# Find P^D(s,t) and add it to the pathway P

    while set(S).intersection(visited) != set(S) or set(T).intersection(visited) != set(T):
        # STOP when S and T are visited
        find_pd(D_path, visited, D, P)
        for node in visited:
            if node not in D.keys() and node not in S and node not in T:
                add_entry_to_D(G, node, D, S, T, D_path)
    return P


if __name__ == "__main__":
    main()
