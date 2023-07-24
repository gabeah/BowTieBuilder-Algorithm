import networkx as nx


def find_pd(D, visited_S, visited_T, S, T, P):
    min_dist = float('inf')
    next_s = None
    next_t = None

    # Step 3: Find the shortest path connecting a 'not visited' node in S to a 'not visited' node in T
    for s in S:
        if s not in visited_S:
            for t in T:
                if t not in visited_T:
                    if D[s][t] < min_dist:
                        min_dist = D[s][t]
                        next_s = s
                        next_t = t

    # Step 4: Add the selected path to P and flag the nodes as 'visited'
    if next_s is not None and next_t is not None:
        visited_S.add(next_s)
        visited_T.add(next_t)
        path = nx.dijkstra_path(G, next_s, next_t, weight='weight')
        for node in path:
            P.add_node(node)
        for i in range(len(path) - 1):
            P.add_edge(path[i], path[i + 1], weight=G[path[i]]
                       [path[i + 1]]['weight'])

    # Step 5: Update D to include all distances to the nodes in PD(s, t)
    for s in S:
        for t in T:
            D[s][t] = min(D[s][t], D[s][next_s] + D[next_s]
                          [next_t] + D[next_t][t])


def bowtie_builder(G, S, T):
    # Step 1: Initialize the pathway P with all nodes S ∩ T, and flag all nodes in S ∩ T as 'not visited'.
    P = nx.Graph()
    common_nodes = set(S) & set(T)
    P.add_nodes_from(common_nodes)
    visited_S = set()
    visited_T = set()

    # Step 2 and Step 6: Calculate the distance matrix D and repeat until every node in S is connected to every node in T in P
    while visited_S != common_nodes or visited_T != common_nodes:
        D = {}
        for s in S:
            D[s] = {}
            for t in T:
                D[s][t] = nx.dijkstra_path_length(G, s, t, weight='weight')

        # Step 3 to Step 5: Find the shortest path and update P and D
        find_pd(D, visited_S, visited_T, S, T, P)

    # Export final pathway P.
    return P


# Example usage:
def main():
   # Create a sample graph]
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

    # Call the Bowtie Builder Algorithm
    bowtie_pathway = bowtie_builder(G, S, T)

    # Print the final pathway P
    print("Final Pathway P:")
    print(bowtie_pathway.edges)
    print(bowtie_pathway.nodes)


def iterate_main_function(iterations):
    for i in range(iterations):
        print(f"Iteration {i + 1}:")
        main()
        print("\n")


if __name__ == "__main__":
    # Call the iterate_main_function to run main() 5 to 10 times
    iterations = 4  # You can change this to 10 if you want 10 iterations
    iterate_main_function(iterations)
