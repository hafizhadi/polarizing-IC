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

# Load .txt and save as gexf
def ConvertTxt(filename):
    graph = LoadGraph(filename)
    print(graph.nodes())
    nx.write_gexf(graph, filename.replace(".txt", ".gexf"))
    return

# Take a networkX graph object and do tons of visualization in NetworkX
def VisualizeGraph(graph):
    plt.figure(figsize=(16, 9))

    # Node plots
    plt.subplot(2,3,1)
    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.annotate("Nodes", xy=(0, 0.5), xycoords=('axes fraction', 'axes fraction'), xytext=(-120,0), textcoords='offset points', size=16)

    nDegree = numpy.array([graph.degree(x) for x in graph.nodes()])
    plt.hist(nDegree, 100)
    plt.axvline(nDegree.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(nDegree.mean()) + "\nsize= " + str(len(nDegree)), xy=(nDegree.mean(), 0.8), xycoords=('data', 'axes fraction'), xytext=(10,0), textcoords='offset points')

    plt.subplot(2,3,2)
    plt.title("Probability Distribution")

    nProb = numpy.array([data["prob"] for node, data in graph.nodes(data = True)])
    plt.hist(nProb, 100)
    plt.axvline(nProb.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(nProb.mean())+ "\nsize=" + str(len(nProb)), xy=(nProb.mean(), 0.8), xycoords=('data', 'axes fraction'), xytext=(10,0), textcoords='offset points')

    plt.subplot(2, 3, 3)
    plt.title("Exposure Count Distribution")

    nCount = numpy.array([data["count"] for node, data in graph.nodes(data=True)])
    plt.hist(nCount, 100)
    plt.axvline(nCount.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(nCount.mean()) + "\nsize=" + str(len(nCount)), xy=(nCount.mean(), 0.8), xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    # Edge plots
    plt.subplot(2, 3, 4)
    plt.ylabel("Count")
    plt.xlabel("# of Degree")
    plt.annotate("Edges", xy=(0, 0.5), xycoords=('axes fraction', 'axes fraction'), xytext=(-120, 0),
                 textcoords='offset points', size=16)

    eDegree = numpy.array([graph.degree(a) + graph.degree(b) for a, b, data in graph.edges(data=True)])
    plt.hist(eDegree, 100)
    plt.axvline(eDegree.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(eDegree.mean())+ "\nsize=" + str(len(eDegree)), xy=(eDegree.mean(), 0.8), xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.subplot(2,3,5)
    plt.xlabel("Diffusion probability")
    eProb = numpy.array([data["prob"] for a, b, data in graph.edges(data=True)])
    plt.hist(eProb, 100)
    plt.axvline(eProb.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(eProb.mean()) + "\nsize=" + str(len(eProb)), xy=(eProb.mean(), 0.8), xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.subplot(2,3,6)
    plt.xlabel("# of Exposure Times")
    eCount = numpy.array([data["count"] for a, b, data in graph.edges(data=True)])
    plt.hist(eCount, 100)
    plt.axvline(eCount.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(eCount.mean()) + "\nsize=" + str(len(eCount)), xy=(eCount.mean(), 0.8), xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.show()

    return

VisualizeGraph(LoadGraph(FILENAME))



