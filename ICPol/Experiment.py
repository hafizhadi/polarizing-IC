import datetime
import json as js
import os
import random
from collections import Counter

import GraphStats as gstats
import networkx as nx
import numpy as np
import scipy.stats as st
import yappi

import ICPol

# Experiment Constants
N_EXP = 1  # Number of individual experiments to do
NODE_VIS_ATTR = ["Degree", "q", "rejCount", "accCount", "propCount"]
EDGE_VIS_ATTR = ["c", "flowCount"]

# Cascade Constants
N_CASCADE = 1001  # Number of cascade
I_DIST = st.uniform(loc=-1, scale=2)
# I_DIST = st.truncnorm(-0.5, 0.5)
# I_DIST = st.beta(0.5, 0.5, loc = -1, scale = 2)

# Periodic snapshot settings
SNAP_MODE = True
SNAP_TIMINGS = np.concatenate((np.arange(1, 100, 10), np.arange(101, 1000, 50), np.arange(1001, 2000, 100),
                               np.arange(2001, 10000, 500), np.arange(10001, 50000, 1000),
                               np.arange(50001, 100000, 5000)))

# Tracking settings
TRACKED_MODE = True
TRACKED_VALUES = ['QMod', 'QModWeak', 'QComm', 'QCommWeak', 'QMean', 'QHom', 'QHist', 'strongOrder', 'strongSize']
TRACK_INTERVAL = 50

# Graph Constants
# List of graph to experiment on
# Array of string between the following: ER, WS, BA, or filename
GRAPH_LIST = [
    '../Datasets/LFR Benchmark Networks/500/0.1/network.dat',
    '../Datasets/LFR Benchmark Networks/500/0.3/network.dat',
    '../Datasets/LFR Benchmark Networks/500/0.5/network.dat',
    'ER', 'BA', 'WS'
]
GRAPH_TYPE = 'ER'
GRAPH_ORDER = 500  # Number of nodes in the graph
pER = 0.04  # Edge probability of the Erdos Renyi random model
nWS = 20  # N parameter of the Watts Strogatz small world model
bWS = 0.2  # Rewiring probability of the Watts Strogatz small world model
nBA = 10  # N parameter of the Barabasi-Albert preferential attachment model


# Return corresponding graph from string
def GetGraph(name):
    try:
        return {
            'ER': nx.erdos_renyi_graph(GRAPH_ORDER, pER),
            'WS': nx.watts_strogatz_graph(GRAPH_ORDER, nWS, bWS),
            'BA': nx.barabasi_albert_graph(GRAPH_ORDER, nBA)
        }[name]
    except KeyError:
        return nx.read_adjlist(name, nodetype=int)


# Return string containing information for the experiment
def ExperimentInfo(icPol, counter):
    expInfo = 'EXPERIMENT INFORMATION\n----------------------\n'
    expInfo += 'Time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n'
    try:
        expInfo += 'Network: ' + str(icPol.graph.order()) + ' nodes ' + str(icPol.graph.size()) + ' edges ' + {
            'ER': ' Erdos Renyi graph w/ p=' + str(pER) + '\n',
            'WS': ' Watts Strogatz graph w/ N=' + str(nWS) + ' and beta=' + str(bWS) + '\n',
            'BA': ' Barabasi Albert graph w/ N=' + str(nBA) + '\n',
        }[GRAPH_LIST[counter]]
    except KeyError:
        expInfo += 'Network: ' + str(icPol.graph.order()) + ' nodes ' + str(icPol.graph.size()) + ' edges from file: ' + \
                   GRAPH_LIST[counter]
    expInfo += str(N_EXP) + ' experiment(2018-04-16 14:38:34#1) of ' + str(N_CASCADE) + ' cascades each\n'
    if (SNAP_MODE): expInfo += 'Snapshotting network on cascade: ' + ','.join(
        str(timing) for timing in SNAP_TIMINGS) + '\n'
    if (TRACKED_MODE): expInfo += 'Tracking ' + ', '.join(TRACKED_VALUES) + ' every ' + str(
        TRACK_INTERVAL) + ' cascade\n'
    expInfo += '\n'

    expInfo += icPol.ModelInfo()
    return expInfo


# Main experiment function
def DoExperiment(nExp, icPol, counter):
    # Container for average experiment data
    totalExperimentData = {
        'individualStats': {
            'nodes': {
                'q': {},
                'rejCount': {},
                'accCount': {},
                'propCount': {},
                'wEdgeHom': {},
                'histQ': {}
            },
            'edges': {
                'c': {},
                'flowCount': {}
            }
        }
    }

    # Make directory for data
    dirname = FILENAME
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(nExp):
        # Container for single experiment data
        trackedVal = dict([(name, []) for name in TRACKED_VALUES])
        cascadeTraceData = {'Trace': []}
        singleExperimentData = {}

        # Do cascades
        for j in range(N_CASCADE):

            print(j + 1)

            # Take a snap of the network between cascades
            if (j + 1 in SNAP_TIMINGS):
                if (SNAP_MODE):
                    nx.write_gexf(icPol.graph, FILENAME + '/' + str(i + 1) + '-' + str(j) + '-Snap ' + '.gexf')

            # Track values between cascades
            if (j % TRACK_INTERVAL == 0):
                if (TRACKED_MODE):
                    # Track Filtered Order and Size
                    if ('strongOrder' in TRACKED_VALUES or 'strongSize' in TRACKED_VALUES):
                        newGraph = icPol.FilterGraphbyConnection(icPol.graph, icPol.CONNECTION_THRESHOLD)

                        if ('strongOrder' in TRACKED_VALUES): trackedVal['strongOrder'].append(newGraph.order())
                        if ('strongSize' in TRACKED_VALUES): trackedVal['strongSize'].append(newGraph.size())

                    # Track measure growth
                    Q = icPol.MeasurePolarizations(TRACKED_VALUES)
                    for name in [i for i in TRACKED_VALUES if i.startswith('Q')]:
                        trackedVal[name].append(Q[name])

            # Cascade
            inf = I_DIST.rvs()
            seedIdx = random.randrange(icPol.graph.order())

            if GRAPH_LIST[counter] not in ['ER', 'BA', 'WS']:
                seedIdx += 1  # Graph index on the LFR Benchmark starts from 1

            cascadeTrace = icPol.Cascade(seedIdx, inf)
            meanEdgeHom = icPol.MeanWeightedEdgeHomogeinity(icPol.graph.edge_subgraph(cascadeTrace))
            cascadeTraceData['Trace'].append((cascadeTrace, meanEdgeHom))
            icPol.Reset()

        # Get experiment data
        singleExperimentData["Result"] = gstats.AnalyzeGraph(icPol.graph)
        singleExperimentData['Result']['TrackedVal'] = trackedVal

        # Aggregate data
        for keys in singleExperimentData["Result"]["individualStats"]["nodes"].keys():
            totalExperimentData["individualStats"]["nodes"][keys] = dict(
                Counter(totalExperimentData["individualStats"]["nodes"][keys]) + Counter(
                    singleExperimentData["Result"]["individualStats"]["nodes"][keys]))

        for keys in singleExperimentData["Result"]["individualStats"]["edges"].keys():
            totalExperimentData["individualStats"]["edges"][keys] = dict(
                Counter(totalExperimentData["individualStats"]["edges"][keys]) + Counter(
                    singleExperimentData["Result"]["individualStats"]["edges"][keys]))

        with open(FILENAME + '/' + str(i + 1) + '-Data ' + ".txt", 'w') as outfile:
            js.dump(singleExperimentData, outfile)

        with open(FILENAME + '/' + str(i + 1) + '-Trace ' + ".txt", 'w') as outfile:
            js.dump(cascadeTraceData, outfile)

        # Make distribution charts
        gstats.VisualizeDistribution(icPol.graph, NODE_VIS_ATTR, EDGE_VIS_ATTR, ExperimentInfo(icPol, counter),
                                     FILENAME + '/' + str(i + 1) + '-Dist' + ".png")

        # Make growth plots
        gstats.VisualizeGrowth(trackedVal, '', FILENAME + '/' + str(i + 1) + '-Growth' + ".png")

        # Clear Data
        icPol.Initialize()

    # Obtain average
    for attr in totalExperimentData["individualStats"]["nodes"].values():
        for vals in attr.values(): vals /= nExp

    for attr in totalExperimentData["individualStats"]["edges"].values():
        for vals in attr.values(): vals /= nExp

    with open(FILENAME + "/Avg.txt", 'w') as outfile:
        js.dump(totalExperimentData, outfile)


# MAIN
for c in range(5):
    ICPol.ICPol.ACC_CONF = 0.5
    ICPol.ICPol.PROP_CONF = c * 0.25

    yappi.start()

    graphs = []
    for i in range(len(GRAPH_LIST)):
        G = GetGraph(GRAPH_LIST[i])
        icPol = ICPol.ICPol(G)
        FILENAME = "Result/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        DoExperiment(N_EXP, icPol, i)
        with open(FILENAME + "/Information.txt", 'w') as outfile:
            outfile.write(ExperimentInfo(icPol, i))
            outfile.write('\nYAPPI LOG\n---------')
            stat = yappi.get_func_stats().print_all(out=outfile)

    yappi.stop()

ICPol.ICPol.C_INC_MODE = 'logistic'

for c in range(5):
    ICPol.ICPol.ACC_CONF = c * 0.25
    ICPol.ICPol.PROP_CONF = 0.5

    yappi.start()

    graphs = []
    for i in range(len(GRAPH_LIST)):
        G = GetGraph(GRAPH_LIST[i])
        icPol = ICPol.ICPol(G)
        FILENAME = "Result/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        DoExperiment(N_EXP, icPol, i)
        with open(FILENAME + "/Information.txt", 'w') as outfile:
            outfile.write(ExperimentInfo(icPol, i))
            outfile.write('\nYAPPI LOG\n---------')
            stat = yappi.get_func_stats().print_all(out=outfile)

    yappi.stop()

for c in range(5):
    ICPol.ICPol.ACC_CONF = c * 0.25
    ICPol.ICPol.PROP_CONF = 1 - (c * 0.25)

    yappi.start()

    graphs = []
    for i in range(len(GRAPH_LIST)):
        G = GetGraph(GRAPH_LIST[i])
        icPol = ICPol.ICPol(G)
        FILENAME = "Result/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        DoExperiment(N_EXP, icPol, i)
        with open(FILENAME + "/Information.txt", 'w') as outfile:
            outfile.write(ExperimentInfo(icPol, i))
            outfile.write('\nYAPPI LOG\n---------')
            stat = yappi.get_func_stats().print_all(out=outfile)

    yappi.stop()

for c in range(5):
    ICPol.ICPol.ACC_CONF = c * 0.25
    ICPol.ICPol.PROP_CONF = c * 0.25

    yappi.start()

    graphs = []
    for i in range(len(GRAPH_LIST)):
        G = GetGraph(GRAPH_LIST[i])
        icPol = ICPol.ICPol(G)
        FILENAME = "Result/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        DoExperiment(N_EXP, icPol, i)
        with open(FILENAME + "/Information.txt", 'w') as outfile:
            outfile.write(ExperimentInfo(icPol, i))
            outfile.write('\nYAPPI LOG\n---------')
            stat = yappi.get_func_stats().print_all(out=outfile)

    yappi.stop()
