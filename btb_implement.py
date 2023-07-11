import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
# from pathlib import Path
# import pandas as pd


def btb_implement(S, T, G):
    # Step 1: Initialize the pathway P with nodes in S ∩ T
    P = nx.Graph()
    common_nodes = set(S) & set(T)
    P.add_nodes_from(common_nodes)

    # visited intially contains 0 things
    visited = set()

    while True:
        # Step 2: Calculate the distance matrix D|S|×|T| between nodes in S and T
        subgraph_S = G.subgraph(S)
        subgraph_T = G.subgraph(T)
        subgraph_nodes = list(subgraph_S.nodes) + list(subgraph_T.nodes)
        node_indices = {node: i for i, node in enumerate(subgraph_nodes)}
        adjacency_matrix = nx.to_scipy_sparse_matrix(
            G, nodelist=subgraph_nodes, weight='weight')
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

        # Step 4: Add nodes and weighted edges of the selected path to P and mark nodes as 'visited' - ERRORS
        edges = [(path[k], path[k+1], {'weight': G[path[k]]
                  [path[k+1]]['weight']}) for k in range(len(path)-1)]
        P.add_edges_from(edges)

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


# Create a sample graph
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 0.5), (1, 3, 0.8), (2, 3, 1.2),
                          (2, 4, 1.0), (3, 5, 0.7), (4, 5, 1.5), (5, 6, 0.9)])

# Define the sets of nodes S and T
S = [1, 2, 3]
T = [4, 5, 6]

# Call the btb_implement function
pathway = btb_implement(S, T, G)

# Print the nodes and edges of the final pathway
print("Pathway Nodes:", pathway.nodes)
print("Pathway Edges:", pathway.edges(data=True))


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


def equation_1(G, pathway, s, t):
    path = pathway[(s, t)][0][0]
    print(path)
    if path == []:
        return float('inf')
    score = 1
    index = 0
    while index < len(path)-1:
        score = score * G[path[index]][path[index+1]]['weight']
        index += 1
    return score


# print(equation_1(G, pathway, 'a', 'f'))
for source in S:
    temp = []
    for target in T:
        temp.append(equation_1(G, pathway, source, target))
    D.append(temp)
