import networkx as nx
from pathlib import Path

"""
Steps:

1. Initialize the pathway P with all nodes S ∩ T, and flag all nodes in S ∩ T as 'not visited'.

2. Calculate the distance matrix D|S|*|T|between the nodes in S and T with Dijkstra's algorithm - determining the maximal scoring (Equation 1 = multiplying the individual confidence values of the utilized edges) paths between the nodes in S and the nodes in T with Dijkstra's algorithm, where the distance is set to ∞ if no path exists. 

3. Select the shortest path in D that connects a 'not visited' and a 'visited' node in P, or, if no such path exists, a 'not visited' node in S to a 'not visited' node in T. 

4. Add the nodes and edges of the selected path to P and flag all nodes in the pathway as 'visited'.

In the next stage of the inference, the highest scoring path PD(s, t) in D that connects a 'not visited' node to a 'visited' node is added. If no such path exists the two 'not visited' nodes with the highest scoring path PD(s, t) in D are connected to each other and, likewise, the path PD(s, t) is added to P.

5. Update D to include all distances to the nodes in PD(s, t).

6. Repeat the steps 2-5 until every node in S is connected to some node in T, and vice versa if such a path exists in G.

Subsequently, the nodes in that path are flagged as 'visited' and D is updated to include all distances to the nodes in PD(s, t). This step is reiterated, in each stage integrating 'not visited' source and target nodes. The method terminates when all nodes in S union T are flagged as 'visited', or, if for the remaining nodes, no path to any other node in S union T exists. Then the final signaling pathway P is returned. 

"""

# Implementation

# functions for reading input files
def read_network(network_file : Path) -> list:
    network = []
    with open(network_file, 'r') as f:
        for line in (f):
            line = line.strip()
            line = line.split('\t')
            network.append((line[0], line[1], float(line[2])))
    return network

def read_source_target(source_file : Path, target_file : Path) -> list:
    source = []
    target = []
    with open(source_file, 'r') as f:
        for line in (f):
            line = line.strip()
            source.append(line)
    with open(target_file, 'r') as f:
        for line in (f):
            line = line.strip()
            target.append(line)
    return source, target

# functions for constructing the network
def construct_network(network : list, source : list, target : list) -> nx.DiGraph:
    Network = nx.DiGraph()
    Network.add_weighted_edges_from(network)
    Network.add_nodes_from(source)
    Network.add_nodes_from(target)
    return Network


def update_D(network : nx.DiGraph, i : str, j : str, D : dict) -> None:
        # check if there is a path between i and j
        if nx.has_path(network, i, j):
            D[(i, j)] = [nx.dijkstra_path_length(network, i, j), nx.dijkstra_path(network, i, j)]
        else:
            D[(i, j)] = [float('inf'), []]
            print(f"There is no path between {i} and {j}")
            
def add_path_to_P(path : list, P : nx.DiGraph) -> None:
        for i in range(len(path) - 1):
            P.add_edge(path[i], path[i + 1])
            
def check_path(network : nx.DiGraph, nodes : list, not_visited : list) -> bool:
    print(f"Nodes: {nodes}")
    print(f"Not visited: {not_visited}")
    for n in not_visited:
        for i in nodes:
            if nx.has_path(network, i, n):
                return True
    return False


def BTB_main(Network : nx.DiGraph, source : list, target : list) -> nx.DiGraph:
    # P is the returned pathway
    P = nx.DiGraph()
    P.add_nodes_from(source)
    P.add_nodes_from(target)

    # Initialize the pathway P with all nodes S union T, and flag all nodes in S union T as 'not visited'.
    not_visited = []
    visited = []
    nodes = list(Network.nodes)

    for i in source:
        not_visited.append(i)
    for j in target:
        not_visited.append(j)
        
    # D is the distance matrix
    D = {}
    for s in source:
        for t in target:
            # If there exists a path between s and t, then calculate the shortest distance between them
            if nx.has_path(Network, s, t):
                D[(s, t)] = [nx.dijkstra_path_length(Network, s, t), nx.dijkstra_path(Network, s, t)]
            else:
                # There is no path between s and t, then set the distance to infinity
                D[(s, t)] = [float('inf'), []]
                print(f"There is no path between {i} and {j}")
                
    print(f'Original D: {D}')

    # source_target is the union of source and target
    source_target = source + target

    # Index is for debugging (will be removed later)
    index = 0
    
    # need to check if there is a path between source and target (not implemented yet)
    while not_visited != []:
        print("\n\nIteration: ", index)
        print(f"Current not visited nodes: {not_visited}")
        
        # Check if there is a path between remaining nodes and not visited nodes
        if not check_path(Network, nodes, not_visited):
            break
                
        # Set initial values
        min_value = float('inf')
        current_path = ()
        current_s = ""
        current_t = ""
        
        # If there is no visited node, then select the shortest path between source and target
        if visited == []:
            # Since now D contains all the shortest paths between source and target, we can just iterate through D to find the shortest path
            for key in D:
                # If the distance is smaller than the current min_value, then update the min_value and the current_path
                if D[key][0] < min_value:
                    min_value = D[key][0]
                    current_path = D[key][1]
                    current_s = key[0]
                    current_t = key[1]


            # Set the distance to infinity
            D[(current_s, current_t)] = [float('inf'), []]
            # Remove the nodes in the current path from not_visited
            not_visited.remove(current_path[0])
            not_visited.remove(current_path[-1])
            # Add the nodes in the current path to visited
            for i in current_path:
                visited.append(i)
                nodes.remove(i)
        else:
            # If there are visited nodes, then select the shortest path between a visited node and a not visited node
            for v in visited:
                for n in not_visited:
                    # Since we don't know if the path is from v to n or from n to v, we need to check both cases
                    if (v, n) in D:
                        if D[(v, n)][0] < min_value:
                            min_value = D[(v, n)][0]
                            current_path = D[(v, n)][1]
                            current_s = v
                            current_t = n
                    if (n, v) in D:
                        if D[(n, v)][0] < min_value:
                            min_value = D[(n, v)][0]
                            current_path = D[(n, v)][1]
                            current_s = n
                            current_t = v

            # Set the distance to infinity
            D[(current_s, current_t)] = [float('inf'), []]
            
            # Add the nodes in the current path to visited
            for i in current_path:
                visited.append(i)
                if i in nodes:
                    nodes.remove(i)
           
            # Remove the nodes in the current path from not_visited
            for i in [current_s, current_t]:
                if i in not_visited:
                    not_visited.remove(i)
                    visited.append(i)
                    
        # Update D
        for i in current_path:
            if i not in source_target: 
                # Since D is a matrix from Source to Target, we need to update the distance from source to i and from i to target
                for s in source:
                    update_D(Network, s, i, D)
                for t in target:
                    update_D(Network, i, t, D)
                # Update the distance from i to i
                D[(i, i)] = [float('inf'), []]
                
        # Add the current path to P
        add_path_to_P(current_path, P)
        
        # some debugging info
        print(f"Min Value: {min_value}")
        print(f"Current selected path: {current_path}")
        print(f"Update D as: {D}")
        print(f"Update visited nodes as: {visited}")
        print(f"Add edges to P as: {P.edges}")
        
        index += 1
    
    print(f"\nThe final pathway is: {P.edges}")
    return P

def write_output(output_file, P):
    with open(output_file, 'w') as f:
        f.write('Node1' + '\t' + 'Node2' + '\n')
        for edge in P.edges:
            f.write(edge[0] + '\t' + edge[1] + '\n')

# -----------Test Case---------------- #
edges = Path('./input/edges2.txt')
source_file = Path('./input/source2.txt')
target_file = Path('./input/target2.txt')
output_file = Path('./output/output2.txt')

network = read_network(edges)
source, target = read_source_target(source_file, target_file)

print(f"Network: {network}")
print(f"Source: {source}")
print(f"Target: {target}")

Network = construct_network(network, source, target)

write_output(output_file, BTB_main(Network, source, target))


