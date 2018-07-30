import networkx as nx
import numpy as np
import yappi
from networkx.algorithms import approximation

# Load Graph
G = nx.barabasi_albert_graph(50, 2)

yappi.start()

# Size/Order
print('Order =  {0}'.format(G.order()))
print('Size = {0}'.format(G.size()))
print('Max Clique = {0}'.format(len(approximation.max_clique(G))))

# Distance
print('Diameter = {0}'.format(nx.diameter(G)))
print('Radius = {0}'.format(nx.radius(G)))
print('Avg Eccentricity = {0}'.format(np.mean([v for k, v in nx.eccentricity(G).items()])))

# Density
print('Density = {0}'.format(nx.density(G)))

# Connectivity
print('Connectivity = {0}'.format(approximation.node_connectivity(G)))

comm = nx.communicability(G)
tComm = sum([sum([v2 for k2, v2 in v1.items()]) for k1, v1 in comm.items()])
print('Communicability  {0}'.format(tComm))

# Efficiency
print('Global Efficiency = {0}'.format(nx.global_efficiency(G)))
print('Local Efficiency = {0}'.format(nx.local_efficiency(G)))

# Path
print('Avg Shortest Path = {0}'.format(nx.average_shortest_path_length(G)))

# Mixing and Assortativity
print('PAssortativity = {0}'.format(nx.degree_pearson_correlation_coefficient(G)))
print('KNNAssortativity = {0}'.format(nx.average_neighbor_degree(G)))

mix = nx.degree_mixing_dict(G)
print('Degree Mixing Matrix = {0}'.format(mix))

# Clustering Coefficient
print('Clustering Coefficient= {0}'.format(approximation.average_clustering(G)))

yappi.get_func_stats().print_all()

yappi.stop()
