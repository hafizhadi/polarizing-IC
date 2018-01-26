import scipy.stats as stats
import random

class IC:
    # Initialize graph attributes for IC process
    @staticmethod
    def Initialize(graph):
        for node, d in graph.nodes_iter(data=True): # Nodes
            d["count"] = 0
            d["infected"] = False

        for n1, n2, d in graph.edges_iter(data=True): # Edges
            d["prob"] = stats.beta.rvs(0.3, 2)
            d["count"] = 0

    # Reset infected
    @staticmethod
    def Reset(graph):
        for node, d in graph.nodes_iter(data=True):  # Nodes
            d['infected'] = False

    # One cascade of the IC model in graph G starting from node a
    @staticmethod
    def Cascade(graph, seed, seedIdx):
        # Activate seed
        seed["infected"] = True
        seed["count"] += 1

        # Do cascade step while a node is still actuve
        nextActives = IC.ICStep(graph, [seedIdx])
        while len(nextActives) > 0:
            nextActives = IC.ICStep(graph, nextActives)


    # One step of an IC cascade in graph G
    @staticmethod
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