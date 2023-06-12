import networkx as nx
from pathlib import Path


# read in source and target sets
source_file = Path("./source.txt")
target_file = Path("./target.txt")

G = nx.Graph()
P = nx.Graph()
# Example with 10 nodes and 15 edges
# In multi_adj: The first label in a line is the source node label followed by the node degree d.
# The next d lines are target node labels and optional edge data.

# 1. Initialize a pathway P with all nodes and flag them as unvisited
elist = [('a', 'b', 1.0), ('a', 'c', 2.0), ('b', 'd', 1.0), ('c', 'e', 1.0),
         ('b', 'e', 5.0), ('b', 'f', 3.0), ('c', 'k',
                                            1.0), ('d', 'e', 2.0), ('d', 'h', 1.0),
         ('d', 'i', 1.0), ('f', 'e', 9.0), ('g', 'e', 1.0), ('e', 'h', 1.0), ('k', 'h', 1.0), ('k', 'i', 1.0), ('k', 'a', 1.0)]
source_set = (['a', 'b', 'c'])
target_set = (['e', 'f', 'g'])
G.add_weighted_edges_from(elist)

# use Dijkstra’s algorithm to find the shortest weighted path between a and f (and repeat for all source and target nodes as well)
# print(nx.dijkstra_path(G, {'a', 'b', 'c'}, {'e', 'f', 'g'}))


print(nx.dijkstra_path(G, 'a', 'f'))
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
print(G.is_directed())

# print all a's neighbors
print(list(G.neighbors('a')))


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
