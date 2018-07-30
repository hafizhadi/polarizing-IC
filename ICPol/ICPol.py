import itertools
import random

import networkx.algorithms.community as comm
import numpy as np
import scipy.stats as st


class ICPol:
    # Constants
    Q_ETA = 0.005  # Volatility of network w.r.t. to Q
    C_ETA = 0.005  # Volatility of network w.r.t. to C

    Q_DIST = st.uniform(loc=-1, scale=2)  # INITIAL DISTRIBUTION OF Q
    C_DIST = st.uniform()  # INITIAL DISTRIBUTION OF C

    POLARIZATION_THRESHOLD = 0.75
    CONNECTION_THRESHOLD = 0.75

    # Dynamics Parameter
    RANGE = 0.15
    ALPHA = 0.5
    CONNECT = 0.1

    # METHODS
    def __init__(self, graph):
        self.graph = graph
        self.Initialize()

    # Return information on model
    def ModelInfo(self):
        info = 'MODEL INFORMATION\n-----------------\n'
        info += 'ALPHA = ' + str(ICPol.ALPHA)
        info += 'RANGE = ' + str(ICPol.RANGE)
        return info

    # CASCADE METHODS

    # Initialize graph attributes for ICPol process
    def Initialize(self):
        for node, d in self.graph.nodes(data=True):
            # Multiple cascade attributes
            d['q'] = np.asscalar(ICPol.Q_DIST.rvs())  # INITIAL DISTRIBUTION OF POLARIZATION
            d['acceptCount'] = 0
            d['rejectCount'] = 0

            # Single cascade attributes
            d['hasProp'] = False  # Propagations status for current cascade
            d['currentExpN'] = 0  # Number of times the node has been exposed to the information of the current cascade

        for n1, n2, d in self.graph.edges(data=True):
            # Multiple cascade attributes
            d['c'] = np.asscalar(ICPol.C_DIST.rvs())  # INITIAL DISTRIBUTION OF CONNECTION STRENGTH
            d['successCount'] = 0
            d['failCount'] = 0

    # Reset node status
    def Reset(self):
        for node, d in self.graph.nodes(data=True):
            d['hasProp'] = False
            d['currentExpN'] = 0

    # One cascade of the ICPol model in graph G starting from seedKey
    # Return a list containing the cascade trace
    def Cascade(self, seedKey, i):
        trace = []

        # First test of seed
        seed = self.graph.node[seedKey]

        if (self.AcceptCheck(seed['q'], i, 0.5)):
            seed['hasProp'] = True

            # Do cascade step while a node is still active
            stepRes = self.ICStep([seedKey], i)
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

        for n in actives:  # Every active nodes propagates information to neighbors
            disconnectList = []

            for neighbor in self.graph.neighbors(n):

                # Neighbors who haven't propagate information considers doing so
                if (not self.graph.node[neighbor]['hasProp']):
                    self.graph.node[neighbor]['hasProp'] = True

                    # TEST FOR ACCEPTANCE
                    if (self.AcceptCheck(self.graph.node[neighbor]['q'], i, self.graph[n][neighbor]['c'])):

                        # Update connection strength
                        self.graph[n][neighbor]['successCount'] += 1
                        self.graph.node[neighbor]['acceptCount'] += 1

                        # MODEL 3
                        if ((abs(self.graph.node[neighbor]['q'] - i) <= ICPol.RANGE)):
                            self.graph.node[neighbor]['q'] += (np.sign(self.graph.node[neighbor]['q']) * ICPol.Q_ETA)
                        else:
                            self.graph.node[neighbor]['q'] += (
                                np.sign(i - self.graph.node[neighbor]['q']) * ICPol.Q_ETA)

                        # self.graph.node[neighbor]['q'] += (np.sign(i - self.graph.node[neighbor]['q']) * ICPol.Q_ETA)

                        self.graph[n][neighbor]['c'] += ICPol.Q_ETA

                        # MODEL 4

                        # # Pick a random non neighbor node from trace
                        # cList = list(set([pair[0] for pair in trace if not self.graph.has_edge(neighbor, pair[0])]))
                        #
                        # if(len(cList) > 0):
                        #     candidate = cList[random.randint(0, len(cList) - 1)]
                        #
                        #     # Connect test
                        #     if(self.ConnectCheck(self.graph.node[neighbor]['q'], self.graph.node[candidate]['q'], self.graph.degree(candidate))):
                        #         # print('ADD {0} {1}'.format(neighbor, candidate))
                        #         self.graph.add_edge(neighbor, candidate, c=0.5, successCount=0, failCount=0)

                        # Add to next actives and trace
                        nextActives.append(neighbor)
                        trace.append((n, neighbor))

                    else:
                        self.graph[n][neighbor]['failCount'] += 1
                        self.graph.node[neighbor]['rejectCount'] += 1

                        # MODEL 2
                        # self.graph.node[neighbor]['q'] -= (np.sign(i - self.graph.node[neighbor]['q']) * ICPol.Q_ETA)

                        self.graph[n][neighbor]['c'] -= ICPol.Q_ETA

                        # # MODEL 4
                        # # Disconnect test
                        # if(self.DisconnectCheck(self.graph.node[n]['q'], self.graph.node[neighbor]['q'], self.graph[n][neighbor]['c'])):
                        #     disconnectList.append(neighbor)

                    self.graph[n][neighbor]['c'] = float(np.clip(self.graph[n][neighbor]['c'], 0, 1))
                    self.graph.node[neighbor]['q'] = float(np.clip(self.graph.node[neighbor]['q'], -1, 1))

                    # for neighbor in disconnectList:
                    #     self.graph.remove_edge(n, neighbor)

        return [nextActives, trace]

    # Check for acceptance
    def AcceptCheck(self, q, i, c):
        return random.random() < (((1 - ICPol.ALPHA) * c + ICPol.ALPHA * (1 - (abs(q - i) / 2))))

    # Check for connection
    def ConnectCheck(self, q, q2, degree):
        return random.random() < ICPol.CONNECT * (1 - (abs(q - q2) / 2)) * (
            degree / max([d for n, d in self.graph.degree()]))

    # Check for disconnection
    def DisconnectCheck(self, q, q2, c):
        return random.random() < ICPol.CONNECT * (abs(q - q2) / 2) * (1 - c)

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

        # Average
        Q = Q / div if div > 0 else 0

        return Q

    # Perform all polarization measure and return result as a dictionary
    def MeasurePolarizations(self, measureList):
        result = {}

        # Filter graph and partition it to two sides accordingly
        strongGraph = self.FilterGraphbyConnection(self.graph, ICPol.CONNECTION_THRESHOLD)
        strongPart = self.PolarizationPartition(strongGraph, ICPol.POLARIZATION_THRESHOLD)

        # Perform

        if ('QMod' in measureList):
            QMod = self.Modularity(strongGraph, strongPart)
            result['QMod'] = QMod

        if ('QComm' in measureList):
            QComm = self.CommBoundary(strongGraph, strongPart)
            result['QComm'] = QComm

        if ('QMean' in measureList):
            QMean = self.MeanAbsPolarization(self.graph)
            result['QMean'] = QMean

        if ('QHom' in measureList):
            QHom = self.MeanWeightedEdgeHomogeinity(self.graph)
            result['QHom'] = QHom

        return result
