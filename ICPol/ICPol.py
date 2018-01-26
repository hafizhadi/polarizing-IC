import random

import numpy as np
import scipy.stats as stats


class IC:
    # Constants
    INCREMENT = 0.01

    # Initialize graph attributes for ICPol process
    @staticmethod
    def Initialize(graph):
        for node, d in graph.nodes_iter(data=True):  # Nodes
            d["q"] = stats.beta.rvs(0.3, 2)  # INITIAL DISTRIBUTION OF POLARIZATION
            d["acc"] = False  # Does node accept current item
            d["accCount"] = 0  # How many time a node has accepted an item
            d["prop"] = False  # Does node propagate current item
            d["propCount"] = 0  # How many time a node has propagated an item

        for n1, n2, d in graph.edges_iter(data=True):  # Edges
            d["p"] = stats.beta.rvs(0.3, 2)  # INITIAL DISTRIBUTION OF CONNECTION STRENGTH
            d["flowCount"] = 0  # How many time an edge has been use for propagation

    # Reset accept and propagate
    @staticmethod
    def Reset(graph):
        for node, d in graph.nodes_iter(data=True):  # Nodes
            d['acc'] = False
            d['prop'] = False

    # One cascade of the ICPol model in graph G starting from node a
    @staticmethod
    def Cascade(graph, seed, seedIdx, i):
        # Start ICPol from seed
        # TOTHINK: Does seed always accept and share?
        seed["acc"] = True
        seed["prop"] = True
        seed["accCount"] += 1
        seed["propCount"] += 1

        # Do cascade step while a node is still active
        nextActives = IC.ICStep(graph, [seedIdx])
        while len(nextActives) > 0:
            nextActives = IC.ICStep(graph, nextActives, i)

    # One step of an ICPol cascade in graph G
    @staticmethod
    def ICStep(graph, actives, i):
        nextActives = []

        for n in actives:
            for neighbor in graph.neighbors_iter(n):
                if (not graph.node[neighbor]["acc"]):  # For all of non-accepting neighbors

                    # TEST FOR ACCEPTANCE
                    if (IC.AccTest(graph.node[neighbor]["q"], i, graph.edge[n][neighbor]["p"])):
                        graph.node[neighbor]["acc"] = True
                        graph.node[neighbor]["accCount"] += 1
                        graph.edge[n][neighbor]["propCount"] += 1

                        # UPDATE ATTRIBUTES FOR ACCEPT
                        graph.node[neighbor]["q"] += np.sign(i - graph.node[neighbor]["q"]) * IC.INCREMENT
                        graph.edge[n][neighbor]["p"] += IC.INCREMENT

                        # TEST FOR PROPAGATION
                        if (IC.PropTest(graph.node[neighbor]["q"], i)):
                            graph.node[neighbor]["prop"] = True
                            graph.node[neighbor]["propCount"] += 1
                            nextActives.append(neighbor)
                    else:
                        # UPDATE ATTRIBUTES FOR NOT ACCEPT
                        graph.node[neighbor]["q"] += np.sign(graph.node[neighbor]["q"]) * IC.INCREMENT
                        graph.edge[n][neighbor]["p"] -= IC.INCREMENT

        return nextActives

    # Formula for acceptance
    @staticmethod
    def AccTest(q, i, p):
        return (random.random() < p * (abs(q - i)))

    # Formula for propagation
    @staticmethod
    def PropTest(q, i):
        return (random.random() < abs(q) * abs(q - i))
