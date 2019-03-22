import copy
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


# Return corresponding graph from string
def GetGraph(name):
    try:
        return {
            'ER': nx.erdos_renyi_graph(GRAPH_ORDER, pER),
            'WS': nx.watts_strogatz_graph(GRAPH_ORDER, nWS, bWS),
            'BA': nx.barabasi_albert_graph(GRAPH_ORDER, nBA),
            'COMPLETE': nx.complete_graph(GRAPH_ORDER)
        }[name]
    except KeyError:
        if (name.endswith('gexf')):
            G = nx.read_gexf(name)
        else:
            G = nx.read_adjlist(name, nodetype=int)

        return G

# Return string containing information for the experiment
def ExperimentInfo(icPol, graphName):
    expInfo = 'EXPERIMENT INFORMATION\n----------------------\n'
    expInfo += 'BETA = ' + str(BETA) + '\n'
    expInfo += 'Time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n'
    try:
        expInfo += 'Network: ' + str(icPol.graph.order()) + ' nodes ' + str(icPol.graph.size()) + ' edges ' + {
            'ER': ' Erdos Renyi graph w/ p=' + str(pER) + '\n',
            'WS': ' Watts Strogatz graph w/ N=' + str(nWS) + ' and beta=' + str(bWS) + '\n',
            'BA': ' Barabasi Albert graph w/ N=' + str(nBA) + '\n',
        }[graphName]
    except KeyError:
        expInfo += 'Network: ' + str(icPol.graph.order()) + ' nodes ' + str(icPol.graph.size()) + ' edges from file: ' + \
                   str(graphName)
    expInfo += str(N_EXP) + ' experiment(2018-04-16 14:38:34#1) of ' + str(N_CASCADE) + ' cascades each\n'
    if (SNAP_MODE): expInfo += 'Snapshotting network on cascade: ' + ','.join(
        str(timing) for timing in SNAP_TIMINGS) + '\n'
    if (TRACK_MODE): expInfo += 'Tracking ' + ', '.join(TRACKED_VALUES) + ' every ' + str(
        TRACK_INTERVAL) + ' cascade\n'
    expInfo += '\n'

    expInfo += icPol.ModelInfo()
    return expInfo

# Dump the result of a single experiment to file
def WriteResults(expData, attrValue, traceData, count):
    # Attribute stats
    with open(FILENAME + '/' + str(count) + '-Data ' + ".txt", 'w') as outfile:
        js.dump(expData, outfile)

    # Cascade stats
    with open(FILENAME + '/' + str(count) + '-Trace ' + ".txt", 'w') as outfile:
        js.dump(traceData, outfile)

    # Distribution histograms
    gstats.VisHistogram(attrValue, FILENAME + '/' + str(count) + '-Dist' + ".png")

    # Growth plots
    gstats.VisGrowth(expData['trackedVal'], '', FILENAME + '/' + str(count) + '-Growth' + ".png")

# Main experiment function
def DoExperiment(nExp, icPol, fromBlank=True, graphs=[]):
    # Variables for averaged data
    totalExpData = copy.deepcopy(DATA_TEMPLATE)
    totalHistData = []
    totalAttrValues = []

    # Make directory for data
    dirname = FILENAME
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Do experiment
    if len(graphs) == 0:
        for i in range(nExp):
            icPol.Initialize()

            # Container for single experiment data
            expData = {
                'analyzedAttr': {},
                'trackedVal': dict([(name, []) for name in TRACKED_VALUES])
            }
            iList = []
            allCascadeData = {
                '#': [],
                'i': [],
                'trace': [],
                'hom': []
            }

            # Do cascades
            for j in range(N_CASCADE):
                print('Exp' + str(i) + ' Cascade ' + str(j + 1))
                icPol.Reset()

                # Cascade
                inf = I_DIST.rvs()  # Opinion score of this cascade's information
                iList.append(inf)
                seedKey = list(icPol.graph.nodes())[random.randrange(len(list(icPol.graph.nodes())))]  # Seed node
                cascadeData = icPol.Cascade(seedKey, inf)

                # Record cascade trace
                allCascadeData['#'].append(j + 1)
                allCascadeData['i'].append(inf)
                allCascadeData['trace'].append(cascadeData)
                # allCascadeData['hom'].append(meanEdgeHom)

                # Take a snap of the network between cascades
                if (j + 1 in SNAP_TIMINGS):
                    if (SNAP_MODE):
                        nx.write_gexf(icPol.graph, FILENAME + '/' + str(i + 1) + '-' + str(j + 1) + '-Snap ' + '.gexf')
                        # attV = gstats.GetValues(icPol.graph, NODE_VIS_ATTR, EDGE_VIS_ATTR)
                        # attV.append(('i', np.asarray(iList)))
                        # gstats.VisHistogram(attV, FILENAME + '/' + str(i + 1) + '-' + str(j + 1) + '-Dist' + ".png")

                # Track values between cascades
                if (((j + 1) == 1) or (((j + 1) % TRACK_INTERVAL) == 0)):
                    if (TRACK_MODE):
                        # Track measure growth
                        Q = icPol.MeasurePolarizations(TRACKED_VALUES)
                        for name in [i for i in TRACKED_VALUES if i.startswith('Q')]:
                            expData['trackedVal'][name].append(Q[name])

                    if ('cascHom' in TRACKED_VALUES):
                        meanEdgeHom = icPol.MeanWeightedEdgeHomogeinity(icPol.graph.edge_subgraph(cascadeData))
                        expData['trackedVal']['cascHom'].append(meanEdgeHom)

                    if ('cascSize' in TRACKED_VALUES):
                        expData['trackedVal']['cascSize'].append(len(cascadeData))

            # Analyze results
            expData['analyzedAttr'] = gstats.AnalyzeAttributes(icPol.graph)
            attrValues = gstats.GetValues(icPol.graph, NODE_VIS_ATTR, EDGE_VIS_ATTR)
            attrValues.append(('i', np.asarray(iList)))
            histData = []

            # Aggregate data
            if (len(totalAttrValues) == 0):
                totalAttrValues = attrValues
            else:
                for atrIdx in range(len(totalAttrValues)):
                    totalAttrValues[atrIdx] = (
                        totalAttrValues[atrIdx][0], np.concatenate((totalAttrValues[atrIdx][1], attrValues[atrIdx][1])))

            if (len(totalHistData) == 0):
                histData = [(name, np.histogram(vals, 25, range=(-1, 1))) if (name == 'q' or name == 'i') else (
                    (name, np.histogram(vals, 25, range=(0, 1))) if name == 'c' else (name, np.histogram(vals, 25))) for
                            name, vals in attrValues]
                totalHistData = histData
            else:
                histData = [()] * len(totalHistData)

                for atrIdx in range(len(totalHistData)):
                    attrHistData = np.histogram(attrValues[atrIdx][1], totalHistData[atrIdx][1][1])
                    histData[atrIdx] = (totalHistData[atrIdx][0], attrHistData)
                    totalHistData[atrIdx] = (totalHistData[atrIdx][0], (
                        [sum(x) for x in zip(totalHistData[atrIdx][1][0], attrHistData[0])], attrHistData[1]))

            for key in totalExpData['trackedVal'].keys():
                if (len(totalExpData['trackedVal'][key]) == 0):
                    totalExpData['trackedVal'][key] = [0] * len(expData['trackedVal'][key])

                totalExpData['trackedVal'][key] = [sum(x) for x in
                                                   zip(totalExpData['trackedVal'][key], expData['trackedVal'][key])]

            for key in totalExpData['analyzedAttr']["nodes"].keys():
                if (len(totalExpData['analyzedAttr']["nodes"][key].keys()) == 0):
                    totalExpData['analyzedAttr']["nodes"][key] = expData['analyzedAttr']['nodes'][key].copy()
                else:
                    count = Counter()
                    count.update(Counter(expData['analyzedAttr']['nodes'][key]))
                    count.update(Counter(totalExpData['analyzedAttr']["nodes"][key]))
                    totalExpData['analyzedAttr']["nodes"][key] = dict(count)

            for key in totalExpData['analyzedAttr']["edges"].keys():
                if (len(totalExpData['analyzedAttr']['edges'][key].keys()) == 0):
                    totalExpData['analyzedAttr']["edges"][key] = expData['analyzedAttr']["edges"][key].copy()
                else:
                    count = Counter()
                    count.update(Counter(expData['analyzedAttr']['edges'][key]))
                    count.update(Counter(totalExpData['analyzedAttr']['edges'][key]))
                    totalExpData['analyzedAttr']['edges'][key] = dict(count)

            # Write results to file
            WriteResults(expData, attrValues, allCascadeData, i + 1)
    else:
        i = 0

        for graph in graphs:
            icPol = ICPol.ICPol(graph, initQC=False)
            icPol.Initialize(initQC=fromBlank)

            # Container for single experiment data
            expData = {
                'analyzedAttr': {},
                'trackedVal': dict([(name, []) for name in TRACKED_VALUES])
            }
            iList = []
            allCascadeData = {
                '#': [],
                'i': [],
                'trace': [],
                'hom': []
            }

            # Do cascades
            for j in range(N_CASCADE):
                print('Exp' + str(i) + ' Cascade ' + str(j + 1))
                icPol.Reset()

                # Cascade
                inf = I_DIST.rvs()  # Opinion score of this cascade's information
                iList.append(inf)
                seedKey = list(icPol.graph.nodes())[random.randrange(len(list(icPol.graph.nodes())))]  # Seed node
                cascadeData = icPol.Cascade(seedKey, inf)

                # Record cascade trace
                allCascadeData['#'].append(j + 1)
                allCascadeData['i'].append(inf)
                allCascadeData['trace'].append(cascadeData)
                # allCascadeData['hom'].append(meanEdgeHom)

                # Take a snap of the network between cascades
                if (j + 1 in SNAP_TIMINGS):
                    if (SNAP_MODE):
                        nx.write_gexf(icPol.graph, FILENAME + '/' + str(i + 1) + '-' + str(j + 1) + '-Snap ' + '.gexf')
                        # attV = gstats.GetValues(icPol.graph, NODE_VIS_ATTR, EDGE_VIS_ATTR)
                        # attV.append(('i', np.asarray(iList)))
                        # gstats.VisHistogram(attV, FILENAME + '/' + str(i + 1) + '-' + str(j + 1) + '-Dist' + ".png")

                # Track values between cascades
                if (((j + 1) == 1) or (((j + 1) % TRACK_INTERVAL) == 0)):
                    if (TRACK_MODE):
                        # Track measure growth
                        Q = icPol.MeasurePolarizations(TRACKED_VALUES)
                        for name in [i for i in TRACKED_VALUES if i.startswith('Q')]:
                            expData['trackedVal'][name].append(Q[name])

                    if ('cascHom' in TRACKED_VALUES):
                        meanEdgeHom = icPol.MeanWeightedEdgeHomogeinity(icPol.graph.edge_subgraph(cascadeData))
                        expData['trackedVal']['cascHom'].append(meanEdgeHom)

                    if ('cascSize' in TRACKED_VALUES):
                        expData['trackedVal']['cascSize'].append(len(cascadeData))

            # Analyze results
            expData['analyzedAttr'] = gstats.AnalyzeAttributes(icPol.graph)
            attrValues = gstats.GetValues(icPol.graph, NODE_VIS_ATTR, EDGE_VIS_ATTR)
            attrValues.append(('i', np.asarray(iList)))
            histData = []

            # Aggregate data
            if (len(totalAttrValues) == 0):
                totalAttrValues = attrValues
            else:
                for atrIdx in range(len(totalAttrValues)):
                    totalAttrValues[atrIdx] = (
                        totalAttrValues[atrIdx][0], np.concatenate((totalAttrValues[atrIdx][1], attrValues[atrIdx][1])))

            if (len(totalHistData) == 0):
                histData = [(name, np.histogram(vals, 25, range=(-1, 1))) if (name == 'q' or name == 'i') else (
                    (name, np.histogram(vals, 25, range=(0, 1))) if name == 'c' else (name, np.histogram(vals, 25))) for
                            name, vals in attrValues]
                totalHistData = histData
            else:
                histData = [()] * len(totalHistData)

                for atrIdx in range(len(totalHistData)):
                    attrHistData = np.histogram(attrValues[atrIdx][1], totalHistData[atrIdx][1][1])
                    histData[atrIdx] = (totalHistData[atrIdx][0], attrHistData)
                    totalHistData[atrIdx] = (totalHistData[atrIdx][0], (
                        [sum(x) for x in zip(totalHistData[atrIdx][1][0], attrHistData[0])], attrHistData[1]))

            for key in totalExpData['trackedVal'].keys():
                if (len(totalExpData['trackedVal'][key]) == 0):
                    totalExpData['trackedVal'][key] = [0] * len(expData['trackedVal'][key])

                totalExpData['trackedVal'][key] = [sum(x) for x in
                                                   zip(totalExpData['trackedVal'][key], expData['trackedVal'][key])]

            for key in totalExpData['analyzedAttr']["nodes"].keys():
                if (len(totalExpData['analyzedAttr']["nodes"][key].keys()) == 0):
                    totalExpData['analyzedAttr']["nodes"][key] = expData['analyzedAttr']['nodes'][key].copy()
                else:
                    count = Counter()
                    count.update(Counter(expData['analyzedAttr']['nodes'][key]))
                    count.update(Counter(totalExpData['analyzedAttr']["nodes"][key]))
                    totalExpData['analyzedAttr']["nodes"][key] = dict(count)

            for key in totalExpData['analyzedAttr']["edges"].keys():
                if (len(totalExpData['analyzedAttr']['edges'][key].keys()) == 0):
                    totalExpData['analyzedAttr']["edges"][key] = expData['analyzedAttr']["edges"][key].copy()
                else:
                    count = Counter()
                    count.update(Counter(expData['analyzedAttr']['edges'][key]))
                    count.update(Counter(totalExpData['analyzedAttr']['edges'][key]))
                    totalExpData['analyzedAttr']['edges'][key] = dict(count)

            # Write results to file
            WriteResults(expData, attrValues, allCascadeData, i + 1)

            i += 1

    # Obtain averaged data
    for i in range(len(totalHistData)):
        totalHistData[i] = (
        totalHistData[i][0], (np.asarray([x / nExp for x in totalHistData[i][1][0]]), totalHistData[i][1][1]))

    for key in totalExpData['trackedVal'].keys():
        totalExpData['trackedVal'][key] = [x / nExp for x in totalExpData['trackedVal'][key]]

    for key in totalExpData['analyzedAttr']['nodes'].keys():
        for iKey, value in totalExpData['analyzedAttr']['nodes'][key].items():
            totalExpData['analyzedAttr']['nodes'][key][iKey] = value / nExp

    for key in totalExpData['analyzedAttr']['edges'].keys():
        for iKey, value in totalExpData['analyzedAttr']['edges'][key].items():
            totalExpData['analyzedAttr']['edges'][key][iKey] = value / nExp

    # Write averaged data to file
    with open(FILENAME + '/Avg-Data.txt', 'w') as outfile:
        js.dump(totalExpData, outfile)

    # Distribution histograms
    gstats.VisHistogram(totalAttrValues, FILENAME + '/Avg-Dist' + ".png", nExp=nExp)

    # Growth plots
    gstats.VisGrowth(totalExpData['trackedVal'], '', FILENAME + '/Avg-Growth' + ".png")


# MAIN

# DEFAULT CONSTANTS

# Snapshot
SNAP_MODE = True
SNAP_TIMINGS = np.concatenate(([1], np.arange(0, 100001, 100)))

# Tracking
TRACK_MODE = True
TRACKED_VALUES = ['QMean', 'QHom', 'cascHom', 'cascSize']
TRACK_INTERVAL = 100

DATA_TEMPLATE = {
    'analyzedAttr': {
        'nodes': {
            'q': {},
            'acceptCount': {},
            'rejectCount': {}
        },
        'edges': {
            'c': {},
            'successCount': {},
            'failCount': {}
        }
    },
    'trackedVal': {
        'QMean': [],
        'QHom': [],
        'cascHom': [],
        'cascSize': []
    }
}

# Visualization
NODE_VIS_ATTR = ['Degree', 'q', 'acceptCount', 'rejectCount']
EDGE_VIS_ATTR = ['c', 'successCount', 'failCount']

# List of graph to experiment on; is a list of strings between the following: ER, WS, BA, or filename
GRAPH_LIST = [
    '../Datasets/LFR Benchmark Networks/250/network.dat'
]

# Graph parameters for graphs generated with networkx
GRAPH_ORDER = 250  # Number of nodes in the graph
pER = 0.08  # Edge probability of the Erdos Renyi random model
nWS = 20  # N parameter of the Watts Strogatz small world model
bWS = 0.2  # Rewiring probability of the Watts Strogatz small world model
nBA = 10  # N parameter of the Barabasi-Albert preferential attachment model

# ALL EXPERIMENT PARAMETERS DETERMINED HERE
N_EXP = 50  # Number of individual experiments to do
N_CASCADE = 10001  # Number of cascade

for bVal in [0.5]:
    BETA = bVal
    I_DIST = st.beta(BETA, BETA, loc=-1, scale=2)

    for cDistVal in [ICPol.ICPol.C_INIT_SCALE]:
        ICPol.ICPol.C_INIT_SCALE = cDistVal
        ICPol.ICPol.C_DIST = st.uniform(scale=cDistVal)

        for qEtaVal in [ICPol.ICPol.Q_ETA]:
            ICPol.ICPol.Q_ETA = qEtaVal

            for cEtaVal in [ICPol.ICPol.C_ETA]:
                ICPol.ICPol.C_ETA = cEtaVal

                for commInitVal in [ICPol.ICPol.COMM_INIT]:
                    ICPol.ICPol.COMM_INIT = commInitVal

                    for qDistVal in [ICPol.ICPol.Q_INIT_BETA]:
                        ICPol.ICPol.Q_INIT_BETA = qDistVal
                        ICPol.ICPol.Q_DIST = st.beta(ICPol.ICPol.Q_INIT_BETA, ICPol.ICPol.Q_INIT_BETA, loc=-1, scale=2)
                        ICPol.ICPol.Q_DIST_HALF = st.beta(1, ICPol.ICPol.Q_INIT_BETA, loc=-1, scale=2)

                        for seedVal in [ICPol.ICPol.SEED_C]:
                            ICPol.ICPol.SEED_C = seedVal

                            for funcVal in ['LINEAR']:
                                ICPol.ICPol.SELECTIVE_FUNCTION = funcVal

                                for ratioVal in ['DEFAULT']:
                                    ICPol.ICPol.SELECTIVE_RATIO = ratioVal

                                    for logMinVal in [ICPol.ICPol.LOG_MIN]:
                                        ICPol.ICPol.LOG_MIN = logMinVal

                                        for logMaxVal in [ICPol.ICPol.LOG_MAX]:
                                            ICPol.ICPol.LOG_MAX = logMaxVal

                                            for logQVal in [ICPol.ICPol.LOG_Q]:
                                                ICPol.ICPol.LOG_Q = logQVal

                                                for logBVal in [ICPol.ICPol.LOG_B]:
                                                    ICPol.ICPol.LOG_B = logBVal

                                                    for logPlusVal in [0.1, 0.2, 0.3, 0.4, 0.5]:
                                                        ICPol.ICPol.LOG_MU = logPlusVal

                                                        for dampVal in [ICPol.ICPol.DAMP]:
                                                            ICPol.ICPol.DAMP = dampVal

                                                            for graphName in GRAPH_LIST:
                                                                G = GetGraph(graphName)
                                                                print(G.order(), G.size())

                                                                yappi.start()
                                                                icPol = ICPol.ICPol(G)
                                                                FILENAME = "Result/" + datetime.datetime.now().strftime(
                                                                    "%Y-%m-%d %H:%M:%S")

                                                                # DO EXPERIMENT
                                                                DoExperiment(N_EXP, icPol)
                                                                with open(FILENAME + "/Information.txt",
                                                                          'w') as outfile:
                                                                    outfile.write(ExperimentInfo(icPol, graphName))
                                                                    outfile.write('\nYAPPI LOG\n---------')
                                                                    stat = yappi.get_func_stats().print_all(out=outfile)

                                                                print(icPol.graph.size())
                                                                yappi.stop()
