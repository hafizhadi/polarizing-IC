import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy

import networkx as nx
import json as js

from collections import Counter
import datetime
import random
from os.path import dirname, abspath

# CONST
FILENAME = dirname(dirname(abspath(__file__))) + "/Preliminary/Result/Exp# 2017-11-21 15:23:45#1000N#1Iter#1000Casc#1.txt"

# Load .txt and return networkX graph object
def LoadGraph(filename):
    graph = nx.Graph()

    with open(filename, 'r') as file:
        jsGraph = js.load(file)

    # Extract nodes
    for key, value in jsGraph["Network"]["Nodes"].items():
        newKey = int(key.replace("'", ""))
        graph.add_node(newKey, value)

    # Extract edges
    for key, value in jsGraph["Network"]["Edges"].items():
        nodes = key.split("-")
        graph.add_edge(int(nodes[0]), int(nodes[1]), value)

    return graph

# Load .txt and save as GraphML
def ConvertTxt(filename):
    graph = LoadGraph(filename)
    print(graph.nodes())
    nx.write_gexf(graph, filename.replace(".txt", ".gexf"))
    return

# Take a networkX graph object and do tons of visualization in NetworkX
def VisualizeGraph(graph):
    return

ConvertTxt(FILENAME)



