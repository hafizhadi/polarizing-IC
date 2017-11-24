import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy

import networkx as nx
import json as js

from collections import Counter
import datetime
import random

import yappi

# CONSTANTS
N = 1000 # Number of nodes in the graph
NExp = 1 # Number of individual experiments to do
NCascade = 1000 # Number of cascade

nBA = 2 # N parameter of the Barabasi-Albert Preferential attachment model

filename ="Result/Exp# " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "#" + str(N) + "N#" + str(NExp) + "Iter#" + str(NCascade) + "Casc#"

# Initialize graph attributes for IC process
def InitializeGraph(graph):
    for node, d in graph.nodes_iter(data=True): # Nodes
        d["count"] = 0
        d["infected"] = False

    for n1, n2, d in graph.edges_iter(data=True): # Edges
        d["prob"] = stats.beta.rvs(0.6, 2)
        d["count"] = 0

# One cascade of the IC model in graph G starting from node a
def ICCascade(graph, seed, seedIdx):
    # Reset infection
    for node, d in graph.nodes_iter(data=True): # Nodes
        d["infected"] = False

    # Activate seed
    seed["infected"] = True
    seed["count"] += 1

    # Do cascade step while a node is still actuve
    nextActives = ICStep(graph, [seedIdx])
    while len(nextActives) > 0:
        nextActives = ICStep(graph, nextActives)

# One step of an IC cascade in graph G
def ICStep(graph, actives):
    nextActives = []

    for n in actives:
        for neighbor in graph.neighbors_iter(n):
            if (not graph.node[neighbor]["infected"]): # For all of non-infected neighbors

                if(random.random() < graph.edge[n][neighbor]["prob"]): # Infect with the diffusion probability
                    graph.node[neighbor]["infected"] = True
                    nextActives.append(neighbor)

                    # Update count
                    graph.node[neighbor]["count"] += 1
                    graph.edge[n][neighbor]["count"] += 1

    return nextActives

# Return statistics of an array of sample
def CalcStats(samples):
    return dict(zip(['mean', 'max', 'min', 'sdeviation'], [float(sp.mean(samples)), float(sp.nanmax(samples)), float(sp.nanmin(samples)), float(sp.std(samples))]))

# Analyze an initialized graph, return a dictionary
def AnalyzeGraph(graph):
    analysisRes = {} # Result

    # Create lists of attributes
    nCount, nProb = [0] * N, [0] * N
    nDCent = list(nx.algorithms.degree_centrality(graph).values())
    nCCent = list(nx.algorithms.closeness_centrality(graph).values())
    nBCent = list(nx.algorithms.betweenness_centrality(graph).values())
    nECent = list(nx.algorithms.degree_centrality(graph).values())
    # nECent = list(nx.algorithms.eigenvector_centrality(graph).values())

    eCount, eProb, eDCent, eCCent, eBCent, eECent = [], [], [], [], [], []

    # Calculate some attributes for the nodes
    for n, d in graph.nodes_iter(data=True):
        nCount[n] = d["count"] # Degree

        probList = [graph[n][d]["prob"] for d in graph[n]]
        if len(probList) == 0: probList = [0]
        meanProb = sum(probList)/len(probList) # Mean edge probability
        nProb[n] = meanProb

        # Add the information to the graph
        graph.node[n]["node"] = graph.degree(n)
        graph.node[n]["prob"] = meanProb

    # Calculate some attributes for the edges
    for a, b, d in graph.edges_iter(data=True):
        eCount.append(d["count"])
        eProb.append(d["prob"])
        eDCent.append(nDCent[a] + nDCent[b])
        eCCent.append(nCCent[a] + nCCent[b])
        eBCent.append(nBCent[a] + nBCent[b])
        eECent.append(nECent[a] + nECent[b])

    # Calculate stats of each list
    nKeys = ["nCount", "nProb", "nDCent", "nCCent", "nBCent", "nECent"]
    nVals = [nCount, nProb, nDCent, nCCent, nBCent, nECent]

    eKeys = ["eCount", "eProb", "eDCent", "eCCent", "eBCent", "eECent"]
    eVals = [eCount, eProb, eDCent, eCCent, eBCent, eECent]
    analysisRes["individualStats"] = {}
    analysisRes["individualStats"]["nodes"] = dict(zip(nKeys, [CalcStats(v) for v in nVals]))
    analysisRes["individualStats"]["edges"] = dict(zip(eKeys, [CalcStats(v) for v in eVals]))

    # Pearson correlation coefficient of count and attributes
    analysisRes["nPearson"] = dict(zip(nKeys, sp.corrcoef(nVals)[0]))
    analysisRes["ePearson"] = dict(zip(eKeys, sp.corrcoef(eVals)[0]))

    return analysisRes

def DoExperiment(nExp):
    totalExperimentData = {
        'individualStats': {
            'nodes': {
                'nCount': {},
                'nProb': {},
                'nDCent': {},
                'nCCent': {},
                'nBCent': {},
                'nECent': {},
            },
            'edges': {
                'eCount': {},
                'eProb': {},
                'eDCent': {},
                'eCCent': {},
                'eBCent': {},
                'eECent': {},
            }
        },
        'ePearson': {
            'eCount': 0,
            'eProb': 0,
            'eDCent': 0,
            'eCCent': 0,
            'eBCent': 0,
            'eECent': 0
        },
        'nPearson': {
            'nCount': 0,
            'nProb': 0,
            'nDCent': 0,
            'nCCent': 0,
            'nBCent': 0,
            'nECent': 0
        }
    }

    for i in range(nExp):
        # Initialize Graph
        # graph = nx.watts_strogatz_graph(N, 10, 0.2)
        # graph = nx.erdos_renyi_graph(N, 0.01)
        graph = nx.barabasi_albert_graph(N, 5)
        InitializeGraph(graph)

        # Do IC process
        for j in range(NCascade):
            index = random.randrange(N)
            ICCascade(graph, graph.node[index], index)

        nx.write_gexf(graph, "Test.gexf")

        # Get experiment data
        singleExperimentData = {}
        singleExperimentData["Network"] = {}
        singleExperimentData["Network"]["Nodes"] = dict([(str(i[0]), i[1]) for i in graph.nodes(data=True)])
        singleExperimentData["Network"]["Edges"] = dict([(str(i[0]) + "-" + str(i[1]), i[2]) for i in graph.edges(data=True)])
        singleExperimentData["Result"] = AnalyzeGraph(graph)

        # Aggregate data
        for keys in singleExperimentData["Result"]["individualStats"]["nodes"].keys():
            totalExperimentData["individualStats"]["nodes"][keys] = dict(Counter(totalExperimentData["individualStats"]["nodes"][keys]) + Counter(singleExperimentData["Result"]["individualStats"]["nodes"][keys]))

        for keys in singleExperimentData["Result"]["individualStats"]["edges"].keys():
            totalExperimentData["individualStats"]["edges"][keys] = dict(Counter(totalExperimentData["individualStats"]["edges"][keys]) + Counter(singleExperimentData["Result"]["individualStats"]["edges"][keys]))

        for k in singleExperimentData["Result"]["ePearson"].keys():
            totalExperimentData["ePearson"][k] += singleExperimentData["Result"]["ePearson"][k]

        for l in singleExperimentData["Result"]["nPearson"].keys():
            totalExperimentData["nPearson"][l] += singleExperimentData["Result"]["nPearson"][l]

        with open(filename + str(i+1) + ".txt", 'w') as outfile:
            js.dump(singleExperimentData, outfile)

        print(i)

        plt.hist([e["prob"] for e in singleExperimentData["Network"]["Edges"].values()], numpy.arange(0,1,0.01))
        plt.show()

    # Obtain average
    for attr in totalExperimentData["individualStats"]["nodes"].values():
        for vals in attr.values(): vals /= nExp

    for attr in totalExperimentData["individualStats"]["edges"].values():
        for vals in attr.values(): vals /= nExp

    for m in totalExperimentData["ePearson"].keys():
        totalExperimentData["ePearson"][m] /= nExp

    for n in totalExperimentData["nPearson"].keys():
        totalExperimentData["nPearson"][n] /= nExp

    with open(filename + "Avg.txt", 'w') as outfile:
        js.dump(totalExperimentData, outfile)

# MAIN
yappi.start()

DoExperiment(NExp)

with open(filename + "Prof.txt", 'w') as outfile:
    stat = yappi.get_func_stats().print_all(out=outfile)
