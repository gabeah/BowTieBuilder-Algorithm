from scipy.sparse.csgraph import dijkstra
import numpy as np
import networkx as nx
from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import argparse


def parse_arguments():
    """
    Process command line arguments.
    @return arguments
    """
    parser = argparse.ArgumentParser(
        description="BowTieBuilder pathway reconstruction"
    )
    parser.add_argument("--network", type=Path, required=True,
                        help="Path to the network file with ',' delimited node pairs")
    parser.add_argument("--source_set", type=Path, required=True,
                        help="Path to the sources file")
    parser.add_argument("--target_set", type=Path, required=True,
                        help="Path to the targets file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to the output file that will be written")

    return parser.parse_args()


def process_input(input_path):
    """
    Process input data with the format 'a', 'b', weight.
    @param input_path: Path to the input file.
    @return: List of tuples containing (source, target, weight).
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split(', ') for line in lines]

    parsed_data = [(source, target, float(weight))
                   for source, target, weight in data]
    return parsed_data


def process_set(input_path):
    """
    Process a set from a file, assuming one element per line.
    @param input_path: Path to the input file.
    @return: List of elements from the file.
    """
    with open(input_path, 'r') as file:
        lines = file.readlines()
        elements = [line.strip() for line in lines]

    return elements


def parse_output_file(output_file: Path):
    with output_file.open() as file:
        nodes = [line.strip() for line in file]

    return nodes


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


def bowtie_builder(network_file: Path, source_set_file: Path, target_set_file: Path, output_file: Path):
    if not network_file.exists():
        raise OSError(f"Network file {str(network_file)} does not exist")
    if not source_set_file.exists():
        raise OSError(f"Source set file {str(source_set_file)} does not exist")
    if not target_set_file.exists():
        raise OSError(f"Target set file {str(target_set_file)} does not exist")
    if output_file.exists():
        print(f"Output files {str(output_file)} will be overwritten")

    # Create the parent directories for the output file if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read the list of nodes in source and target sets
    P = nx.Graph()

    G = nx.Graph()
    with network_file.open() as network_f:
        for line in network_f:
            line = line.strip()
            endpoints, weight = line.split(", ")
            source, target = endpoints.split("|")
            G.add_edge(source, target, weight=float(weight))

    S = process_set(source_set_file)
    T = process_set(target_set_file)

    P.add_nodes_from(set(S) & set(T))
    visited = []
    D_path = {}
    D = {}

    for i in S:
        D_path[i] = {}
        D[i] = {}
        for j in T:
            D_path[i][j] = nx.dijkstra_path(G, i, j, weight='weight')
            D[i][j] = nx.dijkstra_path_length(G, i, j, weight='weight')

    while set(S).intersection(visited) != set(S) or set(T).intersection(visited) != set(T):
        find_pd(D_path, visited, D, P)
        for node in visited:
            if node not in D.keys() and node not in S and node not in T:
                add_entry_to_D(G, node, D, S, T, D_path)

    return P


'''
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
'''

# Example usage
if __name__ == "__main__":

    args = parse_arguments()

    network_file_path = process_input(args.network)
    print("Processed network data:", network_file_path)

    source_set_file_path = process_set(args.source_set)
    target_set_file_path = process_set(args.target_set)

    print("Source set:", source_set_file_path)
    print("Target set:", target_set_file_path)

    output_file_path = Path("path/to/output_file.txt")
    parsed_nodes = parse_output_file(output_file_path)
    print("Parsed nodes:", parsed_nodes)

'''
    network_file_path = Path("path/to/network_file.txt")
    source_set_file_path = Path("path/to/source_set.txt")
    target_set_file_path = Path("path/to/target_set.txt")
    output_file_path = Path("path/to/output_file.txt")
'''
pathway = bowtie_builder(network_file_path, source_set_file_path,
                         target_set_file_path, output_file_path)
print("Pathway:", pathway.edges(data=True))
