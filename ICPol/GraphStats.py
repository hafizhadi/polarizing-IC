import json as js
from os.path import dirname, abspath

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import scipy as sp

# CONST
FILENAME = dirname(
    dirname(abspath(__file__))) + "/Preliminary/Result/Exp# 2017-11-21 15:23:45#1000N#1Iter#1000Casc#1.txt"


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


# Return statistics of an array of sample
def CalcStats(samples):
    return dict(zip(['mean', 'max', 'min', 'sdeviation'],
                    [float(sp.mean(samples)), float(sp.nanmax(samples)), float(sp.nanmin(samples)),
                     float(sp.std(samples))]))


# Analyze an initialized graph, return a dictionary
def AnalyzeGraph(graph):
    analysisRes = {}  # Result

    # Create lists of attributes
    q = []
    accCount = []
    propCount = []

    flowCount = []
    c = []

    # Collect node data
    for n, d in graph.nodes(data=True):
        q.append(d["q"])
        accCount.append(d["accCount"])
        propCount.append(d["propCount"])

        # Add the information to the graph
        graph.node[n]["degree"] = graph.degree(n)

    # Calculate some attributes for the edges
    for a, b, d in graph.edges(data=True):
        flowCount.append(d["flowCount"])
        c.append(d["c"])

        # Calculate stats of each list
    nKeys = ["q", "accCount", "propCount"]
    nVals = [q, accCount, propCount]

    eKeys = ["flowCount", "c"]
    eVals = [flowCount, c]

    analysisRes["individualStats"] = {}
    analysisRes["individualStats"]["nodes"] = dict(zip(nKeys, [CalcStats(v) for v in nVals]))
    analysisRes["individualStats"]["edges"] = dict(zip(eKeys, [CalcStats(v) for v in eVals]))

    return analysisRes


# Take a networkX graph object and do tons of visualization in NetworkX
def VisualizeGraph(graph, filename):
    plt.figure(figsize=(16, 9))

    # Node plots
    plt.subplot(2, 3, 1)
    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.annotate("Nodes", xy=(0, 0.5), xycoords=('axes fraction', 'axes fraction'), xytext=(-120, 0),
                 textcoords='offset points', size=16)

    nDegree = numpy.array([graph.degree(x) for x in graph.nodes()])
    plt.hist(nDegree, 100)
    plt.axvline(nDegree.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(nDegree.mean()) + "\nsize= " + str(len(nDegree)), xy=(nDegree.mean(), 0.8),
                 xycoords=('data', 'axes fraction'), xytext=(10, 0), textcoords='offset points')

    plt.subplot(2, 3, 2)
    plt.title("Acceptance Count Distribution")

    accCount = numpy.array([data["accCount"] for node, data in graph.nodes(data=True)])
    plt.hist(accCount, 100)
    plt.axvline(accCount.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(accCount.mean()) + "\nsize=" + str(len(accCount)), xy=(accCount.mean(), 0.8),
                 xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.subplot(2, 3, 3)
    plt.title("Node Propagation Count Distribution")

    propCount = numpy.array([data["propCount"] for node, data in graph.nodes(data=True)])
    plt.hist(propCount, 100)
    plt.axvline(propCount.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(propCount.mean()) + "\nsize=" + str(len(propCount)), xy=(propCount.mean(), 0.8),
                 xycoords=('data', 'axes fraction'), xytext=(10, 0), textcoords='offset points')

    plt.subplot(2, 3, 4)
    plt.title("Polarization Distribution")

    pol = numpy.array([data["q"] for node, data in graph.nodes(data=True)])
    plt.hist(pol, 100)
    plt.axvline(pol.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(pol.mean()) + "\nsize=" + str(len(pol)), xy=(pol.mean(), 0.8),
                 xycoords=('data', 'axes fraction'), xytext=(10, 0), textcoords='offset points')

    # Edge plots
    plt.subplot(2, 3, 5)
    plt.title("Connection Strength Distribution")

    con = numpy.array([data["c"] for a, b, data in graph.edges(data=True)])
    plt.hist(con, 100)
    plt.axvline(con.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(con.mean()) + "\nsize=" + str(len(con)), xy=(con.mean(), 0.8),
                 xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.subplot(2, 3, 6)
    plt.title("Edge Propagation Count Distribution")
    plt.xlabel("# of Exposure Times")

    flowCount = numpy.array([data["flowCount"] for a, b, data in graph.edges(data=True)])
    plt.hist(flowCount, 100)
    plt.axvline(flowCount.mean(), color='r', linewidth=1, linestyle='dashed')
    plt.annotate("mean= " + str(flowCount.mean()) + "\nsize=" + str(len(flowCount)), xy=(flowCount.mean(), 0.8),
                 xycoords=('data', 'axes fraction'),
                 xytext=(10, 0), textcoords='offset points')

    plt.savefig(filename, dpi='figure')

    return
