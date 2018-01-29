import datetime
import json as js
import random
from collections import Counter

import GraphStats as gstats
import networkx as nx
import scipy.stats as stats
import yappi

import ICPol

# CONSTANTS
N = 500  # Number of nodes in the graph
NExp = 1  # Number of individual experiments to do
NCascade = 50000  # Number of cascade

nBA = 5  # N parameter of the Barabasi-Albert Preferential attachment model

filename = "Result/Exp# " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "#" + str(N) + "N#" + str(
    NExp) + "Iter#" + str(NCascade) + "Casc#"


def DoExperiment(nExp, graph):
    totalExperimentData = {
        'individualStats': {
            'nodes': {
                'q': {},
                'accCount': {},
                'propCount': {}
            },
            'edges': {
                'c': {},
                'flowCount': {}
            }
        }
    }

    # Invoke appropriate cascade object
    exp = ICPol.ICPol()

    for i in range(nExp):
        exp.Initialize(graph)

        # Do IC process
        for j in range(NCascade):
            index = random.randrange(N)
            i = stats.beta.rvs(2, 2)
            exp.Cascade(graph, graph.node[index], index, i)
            exp.Reset(graph)

        # Get experiment data
        singleExperimentData = {}
        singleExperimentData["Network"] = {}
        singleExperimentData["Network"]["Nodes"] = dict([(str(i[0]), i[1]) for i in graph.nodes(data=True)])
        singleExperimentData["Network"]["Edges"] = dict(
            [(str(i[0]) + "-" + str(i[1]), i[2]) for i in graph.edges(data=True)])
        singleExperimentData["Result"] = gstats.AnalyzeGraph(graph)

        # Aggregate data
        for keys in singleExperimentData["Result"]["individualStats"]["nodes"].keys():
            totalExperimentData["individualStats"]["nodes"][keys] = dict(
                Counter(totalExperimentData["individualStats"]["nodes"][keys]) + Counter(
                    singleExperimentData["Result"]["individualStats"]["nodes"][keys]))

        for keys in singleExperimentData["Result"]["individualStats"]["edges"].keys():
            totalExperimentData["individualStats"]["edges"][keys] = dict(
                Counter(totalExperimentData["individualStats"]["edges"][keys]) + Counter(
                    singleExperimentData["Result"]["individualStats"]["edges"][keys]))

        with open(filename + str(i + 1) + ".txt", 'w') as outfile:
            js.dump(singleExperimentData, outfile)

        # Make charts
        gstats.VisualizeGraph(graph, filename + str(i + 1) + ".png")

    # Obtain average
    for attr in totalExperimentData["individualStats"]["nodes"].values():
        for vals in attr.values(): vals /= nExp

    for attr in totalExperimentData["individualStats"]["edges"].values():
        for vals in attr.values(): vals /= nExp

    with open(filename + "Avg.txt", 'w') as outfile:
        js.dump(totalExperimentData, outfile)


# MAIN
yappi.start()

# Initialize Graph
# G = nx.watts_strogatz_graph(N, 10, 0.2)
G = nx.erdos_renyi_graph(N, 0.01)
# G = nx.barabasi_albert_graph(N, 5)

DoExperiment(NExp, G)

with open(filename + "Prof.txt", 'w') as outfile:
    stat = yappi.get_func_stats().print_all(out=outfile)
