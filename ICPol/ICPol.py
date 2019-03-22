import itertools

import networkx.algorithms.community as comm
import numpy as np
import scipy.stats as st


class ICPol:
    # DEFAULT CONSTANTS
    Q_ETA = 0.01  # Volatility of network w.r.t. to Q
    C_ETA = 0.01  # Volatility of network w.r.t. to C

    Q_INIT_BETA = 1
    Q_DIST = st.beta(Q_INIT_BETA, Q_INIT_BETA, loc=-1, scale=2)  # INITIAL DISTRIBUTION OF Q
    C_INIT_SCALE = 0.5
    C_DIST = st.uniform(scale=C_INIT_SCALE)  # INITIAL DISTRIBUTION OF C

    COMM_INIT = False  # Does a community have the same side of opinion?
    Q_DIST_HALF = st.beta(1, 1)  # INITIAL DISTRIBUTION OF Q

    SEED_C = 0.5

    SELECTIVE_FUNCTION = 'LOGISTIC'
    SELECTIVE_RATIO = 'DEFAULT'

    # PARAMETERS FOR THE LOGISTIC FUNCTION
    LOG_MIN = 0
    LOG_MAX = 1
    LOG_Q = 1
    LOG_B = 10  # INDISPENSABLE, NEEDS TO BE > 10
    LOG_MU = 0.15

    DAMP = 1

    POLARIZATION_THRESHOLD = 0.75
    CONNECTION_THRESHOLD = 0.75

    # METHODS
    def __init__(self, graph, initQC=True):
        self.graph = graph
        self.Initialize(initQC)

    # Return information on model
    def ModelInfo(self):
        info = 'MODEL INFORMATION\n-----------------\n'
        info += 'Update Rate of Q:{0} | C:{1}\n'.format(self.Q_ETA, self.C_ETA)
        info += 'Initialized by community? {0}; Initial values of Q:{1} | C:{2}\n'.format(self.COMM_INIT,
                                                                                          self.Q_INIT_BETA,
                                                                                          self.C_INIT_SCALE)
        info += 'Selective Exposure Function: {0}; Parameters: {1} min, {2} max, {3} Q, {4} B, {5} Plus\n'.format(
            self.SELECTIVE_FUNCTION, self.LOG_MIN, self.LOG_MAX, self.LOG_Q, self.LOG_B, self.LOG_MU)
        info += 'Probability damped by {0}\n'.format(self.DAMP)
        info += 'Strong graph threshold Q:{0} | C:{1}\n'.format(self.POLARIZATION_THRESHOLD, self.CONNECTION_THRESHOLD)
        return info

    # CASCADE METHODS

    # Initialize graph attributes for ICPol process
    def Initialize(self, initQC=True):

        for node, d in self.graph.nodes(data=True):
            # Multiple cascade attributes
            if initQC:
                if (ICPol.COMM_INIT):
                    if (d['comm'] % 2 == 1):
                        d['q'] = -np.asscalar(ICPol.Q_DIST_HALF.rvs())  # INITIAL DISTRIBUTION OF POLARIZATION
                    else:
                        d['q'] = np.asscalar(ICPol.Q_DIST_HALF.rvs())  # INITIAL DISTRIBUTION OF POLARIZATION
                else:
                    d['q'] = np.asscalar(ICPol.Q_DIST.rvs())  # INITIAL DISTRIBUTION OF POLARIZATION

            d['acceptCount'] = 0
            d['rejectCount'] = 0

            # Single cascade attributes
            d['hasProp'] = False  # Propagations status for current cascade

        for n1, n2, d in self.graph.edges(data=True):
            # Multiple cascade attributes
            if initQC:
                test = ICPol.C_DIST.rvs()
                d['c'] = np.asscalar(ICPol.C_DIST.rvs())  # INITIAL DISTRIBUTION OF CONNECTION STRENGTH
            d['successCount'] = 0
            d['failCount'] = 0

    # Reset node status
    def Reset(self):
        for node, d in self.graph.nodes(data=True):
            d['hasProp'] = False

    # One cascade of the ICPol model in graph G starting from seedKey
    # Return a list containing the cascade trace
    def Cascade(self, seedKey, i):
        trace = []

        # First test of seed
        seed = self.graph.node[seedKey]

        if (self.AcceptCheck(seed['q'], i, self.SEED_C)):
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
            for neighbor in self.graph.neighbors(n):

                # Neighbors who haven't propagate information considers doing so
                if (not self.graph.node[neighbor]['hasProp']):
                    self.graph.node[neighbor]['hasProp'] = True

                    # TEST FOR ACCEPTANCE
                    if (self.AcceptCheck(self.graph.node[neighbor]['q'], i, self.graph[n][neighbor]['c'])):

                        # Update connection strength
                        self.graph[n][neighbor]['successCount'] += 1
                        self.graph.node[neighbor]['acceptCount'] += 1

                        self.graph.node[neighbor]['q'] = float(max(-1.0, min(1.0,
                                                                             self.graph.node[neighbor]['q'] + np.sign(
                                                                                 i - self.graph.node[neighbor][
                                                                                     'q']) * ICPol.Q_ETA)))
                        self.graph[n][neighbor]['c'] = float(min(1.0, self.graph[n][neighbor]['c'] + ICPol.C_ETA))

                        # Add to next actives and trace
                        nextActives.append(neighbor)
                        trace.append((n, neighbor))

                    else:
                        self.graph[n][neighbor]['failCount'] += 1
                        self.graph.node[neighbor]['rejectCount'] += 1

                        # MODEL 2
                        # self.graph.node[neighbor]['q'] -= (np.sign(i - self.graph.node[neighbor]['q']) * ICPol.Q_ETA)

                        self.graph[n][neighbor]['c'] = float(max(0, self.graph[n][neighbor]['c'] - ICPol.C_ETA))


        return [nextActives, trace]

    # Check for acceptance
    def AcceptCheck(self, q, i, c):
        if (self.SELECTIVE_FUNCTION == 'STEP'):
            difference = 0 if (abs(q - i) / 2) < self.LOG_MU else 1
        elif (self.SELECTIVE_FUNCTION == 'LINEAR'):
            difference = min(max((0, (abs(q - i) / 2) + self.LOG_MU)), 1)
        else:
            difference = self.GenLogistic((abs(q - i) / 2))

        if self.SELECTIVE_RATIO == 'DEFAULT':
            ratio = c  # INDISPENSABLE
        else:
            ratio = self.SELECTIVE_RATIO

        return np.random.random() < (((ratio) * c + (1 - ratio) * (1 - difference))) * self.DAMP

    # Generalized Logistic function with range [0,1]
    def GenLogistic(self, x):
        return self.LOG_MIN + (
            (self.LOG_MAX - self.LOG_MIN) / (1 + ((self.LOG_Q * np.exp(-self.LOG_B * (x - self.LOG_MU))))))


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
