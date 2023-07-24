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

'''
# pass in the targets and sources to know the order, iterate thru the targets list


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
                        print(visited)

        path = D_path[node1][node2]
        P.add_edges_from(zip(path, path[1:]), weight='weight')
        for node in P.nodes:
            if node not in visited:
                visited.append(node)

        print(f"Adding pathway between {node1} and {node2}")
        print(f"The pathway is {path}")
        print(f"The edges in P are {P.edges}")

    print(f"The visited set is: {visited}")


# while set of visited is not set of all nodes in D, add 1 by 1 node in visited set to D
# add the distance of the new node to D, not the node

def add_entry_to_D(G, entry, D, S, T):
    D[entry] = {}
    for i in S:
        D[i][entry] = nx.dijkstra_path_length(G, i, entry, weight='weight')
        for j in T:
            D[entry][j] = nx.dijkstra_path_length(G, entry, j, weight='weight')
    return D


# 6. Repeat steps 2–5 until every node in S is connected to some node in T, and vice versa if such a path exists in G.

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
    unvisited = P.nodes
    visited = []
    print(P)
    print(P.nodes)

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

    while len(S) != 0 or len(T) != 0:
        find_pd(D_path, visited, S, T, D, P)
        print(P.edges)
        print(P.nodes)
        for node in visited:
            if node not in D.keys() and node not in S and node not in T:
                add_entry_to_D(G, node, D, S, T)

# 4. Add the nodes and edges of the selected path to P and flag all nodes in the pathway as 'visited'.

    print(P.edges)

# 5. Update D to include all distances to the nodes in P^D(s, t) (visited??). takes D, return renewed D
# the nodes in that path are flagged as 'visited', The method terminates when all nodes in D are flagged as 'visited'?? ,
# or, if for the remaining nodes, no path to any other node in S∩T exists.

# make wrapper to do 5/10 loops, make sure to see the matrix growing


if __name__ == "__main__":
    main()

'''

def iterate_main_function(iterations):
    for i in range(iterations):
        print(f"Iteration {i + 1}:")
        main()
        print("\n")


if __name__ == "__main__":
    # Call the iterate_main_function to run main() 5 to 10 times
    iterations = 4  # You can change this to 10 if you want 10 iterations
    iterate_main_function(iterations)
'''
