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

# Return statistics of an array of sample
def CalcStats(samples):
    return dict(zip(['mean', 'max', 'min', 'sdeviation'], [float(sp.mean(samples)), float(sp.nanmax(samples)), float(sp.nanmin(samples)), float(sp.std(samples))]))

# Analyze an initialized graph, return a dictionary
def AnalyzeGraph(graph, mode):
    analysisRes = {} # Result

    # Create lists of attributes
    nCount= [0] * graph.order()
    nDCent = list(nx.algorithms.degree_centrality(graph).values())
    nCCent = list(nx.algorithms.closeness_centrality(graph).values())
    nBCent = list(nx.algorithms.betweenness_centrality(graph).values())
    nECent = list(nx.algorithms.degree_centrality(graph).values())
    # nECent = list(nx.algorithms.eigenvector_centrality(graph).values())

    eDCent, eCCent, eBCent, eECent = [], [], [], []

    if(mode == 'IC'):
        nProb = [0] * graph.order()
        eCount = []
        eProb = []
    elif(mode == 'LT'):
        nThres = [0] * graph.order()
        eWeight = []

    # Calculate some attributes for the nodes
    for n, d in graph.nodes_iter(data=True):
        nCount[n] = d["count"] # Degree

        if(mode == 'IC'):
            probList = [graph[n][d]["prob"] for d in graph[n]]
            if len(probList) == 0: probList = [0]
            meanProb = sum(probList)/len(probList) # Mean edge probability
            nProb[n] = meanProb
            graph.node[n]["prob"] = meanProb
        elif(mode == 'LT'):
            nThres[n] = d['threshold'] # Node threshold

        # Add the information to the graph
        graph.node[n]["node"] = graph.degree(n)


    # Calculate some attributes for the edges
    for a, b, d in graph.edges_iter(data=True):
        eDCent.append(nDCent[a] + nDCent[b])
        eCCent.append(nCCent[a] + nCCent[b])
        eBCent.append(nBCent[a] + nBCent[b])
        eECent.append(nECent[a] + nECent[b])

        if(mode == 'IC'):
            eCount.append(d["count"])
            eProb.append(d["prob"])
        elif(mode == 'LT'):
            eWeight.append(d["weight"])

# Calculate stats of each list
    nKeys = ["nCount", "nDCent", "nCCent", "nBCent", "nECent"]
    nVals = [nCount, nDCent, nCCent, nBCent, nECent]

    eKeys = ["eDCent", "eCCent", "eBCent", "eECent"]
    eVals = [eDCent, eCCent, eBCent, eECent]

    if (mode == 'IC'):
        nKeys.append("nProb")
        nVals.append(nProb)
        eKeys.append("eProb")
        eVals.append(eProb)
        eKeys.append("eCount")
        eVals.append(eCount)
    elif (mode == 'LT'):
        nKeys.append("nThres")
        nVals.append(nThres)
        eKeys.append("eWeight")
        eVals.append(eWeight)

    analysisRes["individualStats"] = {}
    analysisRes["individualStats"]["nodes"] = dict(zip(nKeys, [CalcStats(v) for v in nVals]))
    analysisRes["individualStats"]["edges"] = dict(zip(eKeys, [CalcStats(v) for v in eVals]))

    # Pearson correlation coefficient of count and attributes
    analysisRes["nPearson"] = dict(zip(nKeys, sp.corrcoef(nVals)[0]))
    analysisRes["ePearson"] = dict(zip(eKeys, sp.corrcoef(eVals)[0]))

    return analysisRes

# Take a networkX graph object and do tons of visualization in NetworkX
def VisualizeGraph(graph, filename, mode):
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

    plt.subplot(2, 3, 2)
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


    if(mode == 'IC'):
        plt.subplot(2,3,3)
        plt.title("Probability Distribution")

        nProb = numpy.array([data["prob"] for node, data in graph.nodes(data = True)])
        plt.hist(nProb, 100)
        plt.axvline(nProb.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(nProb.mean())+ "\nsize=" + str(len(nProb)), xy=(nProb.mean(), 0.8), xycoords=('data', 'axes fraction'), xytext=(10,0), textcoords='offset points')

        plt.subplot(2,3,4)
        plt.xlabel("Diffusion probability")
        eProb = numpy.array([data["prob"] for a, b, data in graph.edges(data=True)])
        plt.hist(eProb, 100)
        plt.axvline(eProb.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(eProb.mean()) + "\nsize=" + str(len(eProb)), xy=(eProb.mean(), 0.8), xycoords=('data', 'axes fraction'),
                     xytext=(10, 0), textcoords='offset points')

        plt.subplot(2, 3, 5)
        plt.xlabel("# of Exposure Times")
        eCount = numpy.array([data["count"] for a, b, data in graph.edges(data=True)])
        plt.hist(eCount, 100)
        plt.axvline(eCount.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(eCount.mean()) + "\nsize=" + str(len(eCount)), xy=(eCount.mean(), 0.8),
                     xycoords=('data', 'axes fraction'),
                     xytext=(10, 0), textcoords='offset points')

    elif(mode == 'LT'):
        plt.subplot(2, 3, 3)
        plt.title("Threshold Distribution")

        plt.xlabel("Threshold")
        nThres = numpy.array([data["threshold"] for node, data in graph.nodes(data=True)])
        plt.hist(nThres, 100)
        plt.axvline(nThres.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(nThres.mean()) + "\nsize=" + str(len(nThres)), xy=(nThres.mean(), 0.8),
                     xycoords=('data', 'axes fraction'), xytext=(10, 0), textcoords='offset points')

        plt.subplot(2, 3, 6)
        plt.title("Weight Distribution")

        plt.xlabel("Weight")
        eWeight = numpy.array([data["weight"] for a, b, data in graph.edges(data=True)])
        plt.hist(eWeight, 100)
        plt.axvline(eWeight.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(eWeight.mean()) + "\nsize=" + str(len(eWeight)), xy=(eWeight.mean(), 0.8),
                     xycoords=('data', 'axes fraction'),
                     xytext=(10, 0), textcoords='offset points')

    plt.savefig(filename, dpi='figure')

    return



