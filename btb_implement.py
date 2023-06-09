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
         ('b', 'e', 5.0), ('b', 'f', 3.0), ('c', 'k',1.0), ('d', 'e', 2.0), ('d', 'h', 1.0),
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

# for all intermediate nodes, find out edges to target nodes and source nodes

# maybe a better approach would be:
# for e in list of edges
# if e contains a node from the source set and a node from the target set
# run dijkstra_path on (G, source, target)

# approach that makes more sense:

# 1. Initialize a pathway P with all nodes and flag them as unvisited
P_nodes = []
        
print(P_nodes)

P = nx.Graph()
P.add_nodes_from(P_nodes)

# 2. Select a node from the source set and add it to the pathway P
# Multiply confidence values of the utilized edges
pathway = {}
for s in source_set:
    for t in target_set:
        pathway[(s,t)] = []
        pathway[(t,s)] = []
        pathway[(s,t)].append([nx.dijkstra_path(G, s, t), nx.dijkstra_path_length(G, s, t)])
        pathway[(t,s)].append([nx.dijkstra_path(G, t, s), nx.dijkstra_path_length(G, t, s)])

print(pathway)

D = []
def equation_1(G, pathway, s, t):
    path = pathway[(s,t)][0][0]
    print(path)
    if path == []:
        return float('inf')
    score = 1
    index = 0
    while index < len(path)-1:
        score = score * G[path[index]][path[index+1]]['weight']
        index += 1
    return score
    
print(equation_1(G, pathway, 'a', 'f'))
for source in source_set:
    temp = []
    for target in target_set:
        temp.append(equation_1(G, pathway, source, target))
    D.append(temp)
    
print(D)    

# 3. Find P^D(s,t) and add it to the pathway P
visited = []
unvisited = []

for s in source_set:
    unvisited.append(s)
for t in target_set:
    unvisited.append(t)

def find_pd(pathway, visited, unvisited):
    while unvisited != []:
        if visited == []:
            highest = 0

            node1 = ''
            node2 = ''
            for i in range(len(D)):
                for j in range(len(D[i])):
                    if D[i][j] > highest:
                        highest = D[i][j]
                        node1 = source_set[i]
                        node2 = target_set[j]
                        
            index = 0
            while index < len(pathway[(node1,node2)][0][0])-1:
                P.add_edge(pathway[(node1,node2)][0][0][index], pathway[(node1,node2)][0][0][index+1])
                index += 1

            visited.append(node1)
            unvisited.remove(node1)
            visited.append(node2)
            unvisited.remove(node2)
                
            print(f"Adding pathway between {node1} and {node2}")
            print(f"The pathway is {pathway[(node1,node2)][0][0]}")
            print(f"The edges in P are {P.edges}")
        else:
            highest = 0

            node1 = ''
            node2 = ''
            for i in range(len(D)):
                for j in range(len(D[i])):
                    if D[i][j] > highest:
                        if source_set[i] in unvisited and target_set[j] in visited:
                            visited.append(source_set[i])
                            unvisited.remove(source_set[i])
                            highest = D[i][j]
                            node1 = source_set[i]
                            node2 = target_set[j]
                        elif source_set[i] in visited and target_set[j] in unvisited:
                            visited.append(target_set[j])
                            unvisited.remove(target_set[j])
                            highest = D[i][j]
                            node1 = source_set[i]
                            node2 = target_set[j]  
                                           

            index = 0
            while index < len(pathway[(node1,node2)][0][0])-1:
                P.add_edge(pathway[(node1,node2)][0][0][index], pathway[(node1,node2)][0][0][index+1])
                index += 1

            print(f"Adding pathway between {node1} and {node2}")
            print(f"The pathway is {pathway[(node1,node2)][0][0]}")
            print(f"The edges in P are {P.edges}")
    print(visited)
    print(unvisited)

find_pd(pathway, visited, unvisited)
print(P.edges)
print(P.nodes)

'''
1. What is "score" mentioned? Is it just Equation 1?
2. What is the objectives or goals of this algorithm? What is the algorithm trying to maximize or minimize?
3. The structure of the BowTieBuilder algorithm is different from the description (step3). The structure says we are picking the shortest path while the description says we are picking the highest score. Which one is correct?
4. Are they saying finding the maximum scoring path using the shortest path algorithm? If so, how do we do that? Why is Dijsktra's algorithm used? Are we just using the same greedy algorithm to find the maximum scoring path as Dijkstra's algorithm?
4. Step 5: How do we update D based on P?
'''