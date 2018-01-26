import networkx as nx
import json as js

import GraphStats as gstats
import IC
import LT

from collections import Counter
import datetime
import random

import yappi


# CONSTANTS
N = 1000 # Number of nodes in the graph
NExp = 1 # Number of individual experiments to do
NCascade = 1000 # Number of cascade
MODE = 'LT'

nBA = 2 # N parameter of the Barabasi-Albert Preferential attachment model

filename ="Result/Exp# " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "#" + str(N) + "N#" + str(NExp) + "Iter#" + str(NCascade) + "Casc#"

def DoExperiment(nExp, graph, mode):
    totalExperimentData = {
        'individualStats': {
            'nodes': {
                'nCount': {},
                'nProb': {},
                'nDCent': {},
                'nCCent': {},
                'nBCent': {},
                'nECent': {},
                'nThres': {}
            },
            'edges': {
                'eCount': {},
                'eProb': {},
                'eDCent': {},
                'eCCent': {},
                'eBCent': {},
                'eECent': {},
                'eWeight': {}
            }
        },
        'ePearson': {
            'eCount': 0,
            'eProb': 0,
            'eDCent': 0,
            'eCCent': 0,
            'eBCent': 0,
            'eECent': 0,
            'eWeight': 0
        },
        'nPearson': {
            'nCount': 0,
            'nProb': 0,
            'nDCent': 0,
            'nCCent': 0,
            'nBCent': 0,
            'nECent': 0,
            'nThres': 0
        }
    }

    # Invoke appropriate cascade object
    exp = {
        'IC': lambda: IC.IC(),
        'LT': lambda: LT.LT()
    }[mode]()

    for i in range(nExp):
        exp.Initialize(graph)

        # Do IC process
        for j in range(NCascade):
            index = random.randrange(N)
            exp.Cascade(graph, graph.node[index], index)
            exp.Reset(graph)

        # Get experiment data
        singleExperimentData = {}
        singleExperimentData["Network"] = {}
        singleExperimentData["Network"]["Nodes"] = dict([(str(i[0]), i[1]) for i in graph.nodes(data=True)])
        singleExperimentData["Network"]["Edges"] = dict([(str(i[0]) + "-" + str(i[1]), i[2]) for i in graph.edges(data=True)])
        singleExperimentData["Result"] = gstats.AnalyzeGraph(graph, mode)

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

        # Make charts
        gstats.VisualizeGraph(graph, filename + str(i+1) + ".png", mode)

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

# Initialize Graph
G = nx.watts_strogatz_graph(N, 10, 0.2)
# G = nx.erdos_renyi_graph(N, 0.01)
# G = nx.barabasi_albert_graph(N, 5)

DoExperiment(NExp, G, MODE)

with open(filename + "Prof.txt", 'w') as outfile:
    stat = yappi.get_func_stats().print_all(out=outfile)
