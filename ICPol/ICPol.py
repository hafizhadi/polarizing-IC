import itertools
import random

import networkx.algorithms.community as comm
import numpy as np
import scipy.stats as st


class ICPol:
    # Constants
    Q_INCREMENT = 0.01
    Q_INC_MODE = 'none'
    Q_MIN = -0.95
    Q_MAX = 0.95
    Q_DIST = st.uniform(loc=-0.9, scale=1.8)

    C_INCREMENT = 0.01
    C_INC_MODE = 'none'
    C_MIN = 0.05
    C_MAX = 0.95
    C_DIST = st.uniform(loc=0.1, scale=0.9)

    POLARIZATION_THRESHOLD = 0.75
    CONNECTION_THRESHOLD = 0.75

    # Dynamics Parameter
    ACC_CONF = 0.5
    PROP_CONF = 0.5

    # Model Modes
    Q_CLIP = True
    Q_CHANGE_ON_ACCEPT = True
    Q_CHANGE_ON_REJECT = True
    Q_CHANGE_ON_PROPAGATE = False

    C_CLIP = True
    C_CHANGE_ON_ACCEPT = True
    C_CHANGE_ON_REJECT = True
    C_CHANGE_ON_PROPAGATE = False

    # Measures constant
    RWC_SAMPLING_SIZE = 1000
    RWC_PERCENTAGE = 0.25

    # Attributes

    # METHODS
    def __init__(self, graph):
        self.graph = graph
        self.Initialize()

    # Return information on model
    def ModelInfo(self):
        info = 'MODEL INFORMATION\n-----------------\n'
        info += 'Polarization change on accept: ' + str(ICPol.Q_CHANGE_ON_ACCEPT) + '| on reject: ' + str(
            ICPol.Q_CHANGE_ON_REJECT) + '| on propagate: ' + str(ICPol.Q_CHANGE_ON_PROPAGATE) + '\n'
        info += 'Change = ' + ICPol.Q_INC_MODE + ' decreasing (' + str(ICPol.Q_INCREMENT) + ')\n'
        info += 'Clipping for polarization: ' + str(ICPol.Q_CLIP) + '(' + str(ICPol.Q_MIN) + ', ' + str(
            ICPol.Q_MAX) + ')\n\n'

        info += 'Connection strength change on accept: ' + str(ICPol.C_CHANGE_ON_ACCEPT) + '| on reject: ' + str(
            ICPol.C_CHANGE_ON_REJECT) + '| on propagate: ' + str(ICPol.C_CHANGE_ON_PROPAGATE) + '\n'
        info += 'Change = ' + ICPol.C_INC_MODE + ' decreasing (' + str(ICPol.C_INCREMENT) + ')\n'
        info += 'Clipping for connection: ' + str(ICPol.C_CLIP) + '(' + str(ICPol.C_MIN) + ', ' + str(
            ICPol.C_MAX) + ')\n\n'

        info += 'Polarization Threshold: ' + str(ICPol.POLARIZATION_THRESHOLD) + '\n'
        info += 'Connection Threshold: ' + str(ICPol.CONNECTION_THRESHOLD) + '\n'

        return info

    # Return a snapshot of the network as a Dictionary object
    # UNUSED
    def NetworkSnap(self):
        snap = {}
        snap["Nodes"] = dict(
            [(str(i), {k: v for k, v in d.items() if not isinstance(v, bool)}) for i, d in self.graph.nodes(data=True)])
        snap["Edges"] = dict(
            [(str(i[0]) + "-" + str(i[1]), i[2]) for i in self.graph.edges(data=True)])

        return snap

    # CASCADE METHODS
    # Initialize graph attributes for ICPol process
    def Initialize(self):
        for node, d in self.graph.nodes(data=True):  # Nodes
            d['q'] = np.asscalar(ICPol.Q_DIST.rvs())  # INITIAL DISTRIBUTION OF POLARIZATION
            d['rejCount'] = 0  # How many time a node has accepted an item
            d['acc'] = False  # Does node accept current item
            d['accCount'] = 0  # How many time a node has accepted an item
            d['prop'] = False  # Does node propagate current item
            d['propCount'] = 0  # How many time a node has propagated an item

            # Polarization Measures
            d['wEdgeHom'] = 0  # Edge Homogeinity of the node
            d['histQ'] = 0  # History Polarization of the node

        for n1, n2, d in self.graph.edges(data=True):  # Edges
            d['c'] = np.asscalar(ICPol.C_DIST.rvs())  # INITIAL DISTRIBUTION OF CONNECTION STRENGTH
            d['flowCount'] = 0  # How many time an edge has been use for propagation

    # Reset accept and propagate
    def Reset(self):
        for node, d in self.graph.nodes(data=True):  # Nodes
            d['acc'] = False
            d['prop'] = False

    # One cascade of the ICPol model in graph G starting from node a
    # RETURN A LIST CONTAINING THE CASCADE TRACE
    def Cascade(self, seedIdx, i):
        trace = []

        # Start ICPol from seed
        seed = self.graph.node[seedIdx]

        seed['acc'] = True
        seed['prop'] = True
        seed['rejCount'] += 1
        seed['accCount'] += 1
        seed['propCount'] += 1

        # Do cascade step while a node is still active
        stepRes = self.ICStep([seedIdx], i)
        nextActives = stepRes[0]
        trace = trace + stepRes[1]

        while len(nextActives) > 0:
            stepRes = self.ICStep(nextActives, i)
            nextActives = stepRes[0]
            trace = trace + stepRes[1]

        return trace

    # One step of an ICPol cascade in graph G
    def ICStep(self, actives, i):
        nextActives = []
        trace = []

        for n in actives:
            for neighbor in self.graph.neighbors(n):
                if (not self.graph.node[neighbor]['acc']):  # For all of non-accepting neighbors

                    # Calculate Increment
                    if (self.graph.node[neighbor]['q'] != 0):
                        IncQ = ICPol.Q_INCREMENT * {'linear': 1 - abs(self.graph.node[neighbor]['q']),
                                                    'logistic': 1 - self.GenLogistic(
                                                        abs(self.graph.node[neighbor]['q']), 1000, 12),
                                                    'none': 1}[ICPol.Q_INC_MODE]
                    else:
                        IncQ = ICPol.Q_INCREMENT

                    if self.graph[n][neighbor]['c'] != 0:
                        IncC = ICPol.C_INCREMENT * {'linear': 1 - self.graph[n][neighbor]['c'],
                                                    'logistic': 1 - self.GenLogistic(abs(self.graph[n][neighbor]['c']),
                                                                                     1000, 10),
                                                    'none': 1}[ICPol.C_INC_MODE]
                    else:
                        IncC = ICPol.C_INCREMENT


                    # TEST FOR ACCEPTANCE
                    if (self.AccTest(self.graph.node[neighbor]['q'], i, self.graph[n][neighbor]['c'])):
                        self.graph.node[neighbor]['acc'] = True
                        self.graph.node[neighbor]['accCount'] += 1
                        self.graph[n][neighbor]['flowCount'] += 1

                        # UPDATE ATTRIBUTES FOR ACCEPT
                        if (ICPol.Q_CHANGE_ON_ACCEPT): self.graph.node[neighbor]['q'] += (
                            np.sign(i - self.graph.node[neighbor]['q']) * IncQ)
                        if (ICPol.C_CHANGE_ON_ACCEPT): self.graph[n][neighbor]['c'] += IncC

                        # SAVE CASCADE TRACE
                        trace.append((n, neighbor))

                        # TEST FOR PROPAGATION
                        if (self.PropTest(self.graph.node[neighbor]['q'], i)):
                            self.graph.node[neighbor]['prop'] = True
                            self.graph.node[neighbor]['propCount'] += 1
                            nextActives.append(neighbor)
                    else:
                        # UPDATE ATTRIBUTES FOR NOT ACCEPT
                        self.graph.node[neighbor]['rejCount'] += 1
                        if (ICPol.Q_CHANGE_ON_REJECT): self.graph.node[neighbor]['q'] += (
                            np.sign(self.graph.node[neighbor]['q']) * IncQ)
                        if (ICPol.C_CHANGE_ON_REJECT): self.graph[n][neighbor]['c'] -= (IncC)

                # Clip
                if (ICPol.Q_CLIP): self.graph.node[neighbor]['q'] = np.clip(self.graph.node[neighbor]['q'], ICPol.Q_MIN,
                                                                            ICPol.Q_MAX)
                if (ICPol.C_CLIP): self.graph[n][neighbor]['c'] = np.clip(self.graph[n][neighbor]['c'], ICPol.C_MIN,
                                                                          ICPol.C_MAX)

                # Scalarize
                self.graph.node[neighbor]['q'] = np.asscalar(self.graph.node[neighbor]['q'])
                self.graph[n][neighbor]['c'] = np.asscalar(self.graph[n][neighbor]['c'])

        return [nextActives, trace]

    # Formula for acceptance
    def AccTest(self, q, i, c):
        return (random.random() < ((1 - ICPol.ACC_CONF) * c + ICPol.ACC_CONF * (1 - (abs(q - i) / 2))))

    # Formula for propagation
    def PropTest(self, q, i):
        return (random.random() < ((1 - ICPol.PROP_CONF) * abs(q) + ICPol.PROP_CONF * (1 - (abs(q - i) / 2))))

    # Generalized Logistic function with range [0,1]
    def GenLogistic(self, x, Q, B):
        return 1 / (1 + (Q * np.exp(-B * x)))

    # POLARIZATION MEASURE METHODS

    # Output partition of polarization (q < -threshold and q > threshold)
    # RETURN [[> THREHSOLD NODES], [< -THRESHOLD NODES], [THE REST]]
    def PolarizationPartition(self, g, threshold):
        # Initialize list
        a = []
        b = []
        c = []

        # Iterate the nodes
        for node, d in g.nodes(data=True):
            if (d['q'] > threshold):
                a.append(node)
            elif (d['q'] < -threshold):
                b.append(node)
            else:
                c.append(node)
        return [a, b, c]

    # Output graph that is constructed only from edges with (c > threshold)
    def FilterGraphbyConnection(self, g, threshold):
        strongEdges = [(a, b) for a, b, d in g.edges(data=True) if d['c'] > threshold]
        newGraph = g.edge_subgraph(strongEdges)

        return newGraph

    # Modularity
    def Modularity(self, graph, part):
        return comm.modularity(graph, communities=part)

    # Community Boundary (Guerra et al, 2017)
    # Note that this implementation is limited to two group partition
    # Also it's annoyingly inefficient as hell: TODO
    def CommBoundary(self, graph, part):
        # Find Candidates for boundary nodes
        cBoundA = []
        cBoundB = []
        for n1, n2 in graph.edges(data=False):
            if (n1 in part[0]):
                if (n2 in part[1]):
                    if n1 not in cBoundA: cBoundA.append(n1)
                    if n2 not in cBoundB: cBoundB.append(n2)
            else:
                if (n2 in part[0]):
                    if n1 not in cBoundB: cBoundB.append(n1)
                    if n2 not in cBoundA: cBoundA.append(n2)

        # If no boundary nodes exist, the two partition is truly separate
        if len(cBoundA) == 0 and len(cBoundB) == 0:
            return 1

        # Internal nodes
        intA = [x for x in part[0] if x not in cBoundA]
        intB = [x for x in part[1] if x not in cBoundB]

        # Find boundary nodes
        boundA = []
        boundB = []
        for cand in cBoundA:
            for node in intA:
                if (graph.has_edge(cand, node)):
                    boundA.append(cand)

        for cand in cBoundB:
            for node in intB:
                if (graph.has_edge(cand, node)):
                    boundB.append(cand)

        # Calculate polarization
        Q = 0

        # Iterate boundary nodes for partition A
        for node in boundA:
            intEdge = sum(1 for pair in itertools.product([node], intA) if self.graph.has_edge(pair[0], pair[1]))
            crossEdge = sum(1 for pair in itertools.product([node], boundB) if self.graph.has_edge(pair[0], pair[1]))
            Q += ((intEdge / (intEdge + crossEdge)) - 0.5)

        # Iterate boundary nodes for partition B
        for node in boundB:
            intEdge = sum(1 for pair in itertools.product([node], intB) if self.graph.has_edge(pair[0], pair[1]))
            crossEdge = sum(1 for pair in itertools.product([node], boundA) if self.graph.has_edge(pair[0], pair[1]))
            Q += ((intEdge / (intEdge + crossEdge)) - 0.5)

        # Divide by the number of boundary nodes
        Q = Q / (len(boundA) + len(boundB)) if (len(boundA) + len(boundB) > 0) else 0

        return Q

    # Regular mean absolute polarization
    def MeanAbsPolarization(self, graph):
        Q = 0

        for n, d in graph.nodes(data=True):
            Q += abs(d['q'])

        Q = Q / graph.order()

        return Q

    # Weighted Mean Edge Homogeneity
    # Return mean polarization while appending individual polarization to each node
    def MeanWeightedEdgeHomogeinity(self, graph):
        Q = sum([(graph.node[n1]['q'] * graph.node[n2]['q'] * d['c']) for n1, n2, d in graph.edges(data=True)])
        div = sum([d['c'] for n1, n2, d in graph.edges(data=True)])

        # # Calculate homogeinity and append to every node the sum of its edge homogeneity
        # for n1, n2, d in graph.edges(data = True):
        #     eQ = (graph.node[n1]['q'] * graph.node[n2]['q'] * d['c']) # Calculate
        #     d['wEdgeHom'] = eQ
        #     graph.node[n1]['wEdgeHom'] += eQ
        #
        #     Q += eQ # Sum of all homogeneity
        #     div += d['c'] # Sum of all connection strength
        #
        #
        # # Calculate average homogeinity for every node
        # for n, d in graph.nodes(data = True):
        #     d['wEdgeHom'] = d['wEdgeHom'] / sum([graph[n][d]['c'] for d in graph.neighbors(n)]) if(graph.degree(n) > 0) else 0

        # Average
        Q = Q / div if div > 0 else 0

        return Q

    # Polarization score from interaction history
    # Return mean polarization while appending individual polarization to each node
    def HistoryPolarization(self, graph):
        Q = 0

        # Calculate and append history Polarization to every node
        for n, d in graph.nodes(data=True):
            pos = 0
            neg = 0

            for neighbor in graph.neighbors(n):
                if (graph.node[neighbor]['q'] >= 0):
                    pos += graph[n][neighbor]['flowCount']
                else:
                    neg += graph[n][neighbor]['flowCount']

            # POLARIZATION FORMULA (POS - NEG) / (POS + NEG)
            nQ = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
            d['histQ'] = nQ
            Q += abs(nQ)

        # Average
        Q = Q / graph.order()

        return Q

    # RWC Score Methods
    # Do a random walk, return the side where it ends
    def RandomWalk(self, graph, start, goals):
        # Variables
        current = start
        goal = False
        side = ''

        while (not goal):
            # Walk to random neighbor
            neighbors = graph.neighbors(current)
            current = list(neighbors)[random.randint(0, graph.degree(current) - 1)]

            # Check if in goal
            if (current in goals[0]):
                side = 'LEFT'
                goal = True
            if (current in goals[1]):
                side = 'RIGHT'
                goal = True

        return side

    # Main method
    def RWC(self, graph, partition):
        Q = 0

        # Counter for random walks: start_end
        left_left = 0
        left_right = 0
        right_right = 0
        right_left = 0

        # Do Monte Carlo sampling
        for iter in range(ICPol.RWC_SAMPLING_SIZE):

            # Get list of goal nodes from both sides
            nodes = [[], []]
            for i in range(2):
                # For the allocated percentage of nodes in the partition
                for j in range(int(len(partition[i]) * ICPol.RWC_PERCENTAGE)):
                    # Append a random node
                    nodes[i].append(partition[i][random.randint(0, len(partition[i]) - 1)])

            # Start a random walk for every nodes in both sides
            for i in range(2):
                for j in nodes[i]:
                    end0 = self.RandomWalk(graph, j, nodes)

                    # Record where it ends
                    if (j in partition[0]):
                        if (end0 == 'LEFT'):
                            left_left += 1
                        else:
                            left_right += 1
                    else:
                        if (end0 == 'LEFT'):
                            right_left += 1
                        else:
                            right_right += 1

        # Calculate RWC
        e1 = left_left * 1.0 / (left_left + right_left) if (left_left + right_left) > 0 else 0
        e2 = left_right * 1.0 / (left_right + right_right) if (left_right + right_right) > 0 else 0
        e3 = right_left * 1.0 / (left_left + right_left) if (left_left + right_left) > 0 else 0
        e4 = right_right * 1.0 / (left_right + right_right) if (left_right + right_right) > 0 else 0
        Q = e1 * e4 - e2 * e3

        return Q

    # Perform all polarization measure and return result as a dictionary
    def MeasurePolarizations(self, measureList):
        result = {}

        # Filter graph and partition it to two sides accordingly
        strongGraph = self.FilterGraphbyConnection(self.graph, ICPol.CONNECTION_THRESHOLD)
        strongPart = self.PolarizationPartition(strongGraph, ICPol.POLARIZATION_THRESHOLD)

        # Weak partition for some of the measure
        # Randomly distributes nodes with 0 polarization so all nodes are included to two parts
        weakPart = self.PolarizationPartition(self.graph, 0)
        if (len(weakPart[2]) > 0):
            for node in weakPart[2]:
                if (random.random() > 0.5):
                    weakPart[0].append(node)
                else:
                    weakPart[1].append(node)

            weakPart = [weakPart[0], weakPart[1]]

        # Perform
        if ('QRWC' in measureList):
            QRWC = self.RWC(strongGraph, strongPart)
            result['QRWC'] = QRWC

        if ('QRWCWeak' in measureList):
            QRWCWeak = self.RWC(self.graph, weakPart)
            result['QRWCWeak'] = QRWCWeak

        if ('QMod' in measureList):
            QMod = self.Modularity(strongGraph, strongPart)
            result['QMod'] = QMod

        if ('QModWeak' in measureList):
            QModWeak = self.Modularity(self.graph, weakPart)
            result['QModWeak'] = QModWeak

        if ('QComm' in measureList):
            QComm = self.CommBoundary(strongGraph, strongPart)
            result['QComm'] = QComm

        if ('QCommWeak' in measureList):
            QCommWeak = self.CommBoundary(self.graph, weakPart)
            result['QCommWeak'] = QCommWeak

        if ('QMean' in measureList):
            QMean = self.MeanAbsPolarization(self.graph)
            result['QMean'] = QMean

        if ('QHom' in measureList):
            QHom = self.MeanWeightedEdgeHomogeinity(self.graph)
            result['QHom'] = QHom

        if ('QHist' in measureList):
            QHist = self.HistoryPolarization(self.graph)
            result['QHist'] = QHist

        return result
