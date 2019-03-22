import networkx as nx

GRAPH_LIST = [
    '../Datasets/Processed/POLARIZED/baltimore/baltimore-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/beefban/beefban-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/gunsense/gunsense-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/indiana/indiana-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/indiasdaughter/indiasdaughter-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/leadersdebate/leadersdebate-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/nemtsov/nemtsov-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/netanyahu/netanyahu-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/russia_march/russia_march-PROCESSED.txt',
    '../Datasets/Processed/POLARIZED/ukraine/ukraine-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/ultralive-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/germanwings-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/mothersday-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/onedirection-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/nepal-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/jurassicworld-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/ff-PROCESSED.txt',
    '../Datasets/Processed/NOT POLARIZED/sxsw-PROCESSED.txt',
]

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Nodes: {}'.format(G.order()))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Edges: {}'.format(G.size()))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Density: {}'.format(nx.density(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Average shortest path: {}'.format(nx.average_shortest_path_length(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Assortativity: {}'.format(nx.degree_assortativity_coefficient(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Average Degree Connectivity: {}'.format(nx.average_degree_connectivity(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Clustering: {}'.format(nx.average_clustering(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Transitivity: {}'.format(nx.transitivity(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Diameter: {}'.format(nx.diameter(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Radius: {}'.format(nx.radius(G)))

for graph in GRAPH_LIST:
    G = nx.read_adjlist(graph, nodetype=int)
    print(graph)
    print('Transitivity: {}'.format(nx.transitivity(G)))
