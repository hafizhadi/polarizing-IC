import random

import numpy as np


class ICPol:
    # Constants
    INCREMENT = 0.01

    # Initialize graph attributes for ICPol process
    @staticmethod
    def Initialize(graph):
        for node, d in graph.nodes(data=True):  # Nodes
            d["q"] = random.random()  # INITIAL DISTRIBUTION OF POLARIZATION
            d["acc"] = False  # Does node accept current item
            d["accCount"] = 0  # How many time a node has accepted an item
            d["prop"] = False  # Does node propagate current item
            d["propCount"] = 0  # How many time a node has propagated an item

        for n1, n2, d in graph.edges(data=True):  # Edges
            d["c"] = random.random()  # INITIAL DISTRIBUTION OF CONNECTION STRENGTH
            d["flowCount"] = 0  # How many time an edge has been use for propagation

    # Reset accept and propagate
    @staticmethod
    def Reset(graph):
        for node, d in graph.nodes(data=True):  # Nodes
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
        nextActives = ICPol.ICStep(graph, [seedIdx], i)
        while len(nextActives) > 0:
            nextActives = ICPol.ICStep(graph, nextActives, i)

    # One step of an ICPol cascade in graph G
    @staticmethod
    def ICStep(graph, actives, i):
        nextActives = []

        for n in actives:
            for neighbor in graph.neighbors(n):
                if (not graph.node[neighbor]["acc"]):  # For all of non-accepting neighbors

                    # TEST FOR ACCEPTANCE
                    if (ICPol.AccTest(graph.node[neighbor]["q"], i, graph[n][neighbor]["c"])):
                        graph.node[neighbor]["acc"] = True
                        graph.node[neighbor]["accCount"] += 1
                        graph[n][neighbor]["flowCount"] += 1

                        # UPDATE ATTRIBUTES FOR ACCEPT
                        graph.node[neighbor]["q"] = np.clip(
                            graph.node[neighbor]["q"] + (np.sign(i - graph.node[neighbor]["q"]) * ICPol.INCREMENT), 0.1,
                            0.9)
                        graph[n][neighbor]["c"] = np.clip(graph[n][neighbor]["c"] + ICPol.INCREMENT, 0, 1)

                        # TEST FOR PROPAGATION
                        if (ICPol.PropTest(graph.node[neighbor]["q"], i)):
                            graph.node[neighbor]["prop"] = True
                            graph.node[neighbor]["propCount"] += 1
                            nextActives.append(neighbor)
                    else:
                        # UPDATE ATTRIBUTES FOR NOT ACCEPT
                        graph.node[neighbor]["q"] = np.clip(
                            graph.node[neighbor]["q"] + (np.sign(graph.node[neighbor]["q"] - 0.5) * 0), 0.1, 0.9)
                        graph[n][neighbor]["c"] = np.clip(graph[n][neighbor]["c"] - ICPol.INCREMENT, 0, 1)

        return nextActives

    # Formula for acceptance
    @staticmethod
    def AccTest(q, i, c):
        return (random.random() < c * (abs(q - i)))

    # Formula for propagation
    @staticmethod
    def PropTest(q, i):
        return (random.random() < abs(q) * abs(q - i))
