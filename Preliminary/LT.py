import scipy.stats as stats

class LT():
    # Initialize graph attributes for LT process
    @staticmethod
    def Initialize(graph):
        for node, d in graph.nodes_iter(data=True): # Nodes
            d['count'] = 0
            d['infected'] = False
            d['threshold'] = stats.uniform.rvs()

        for n1, n2, d in graph.edges_iter(data=True): # Edges
            d['weight'] = stats.beta.rvs(1,2)

    # Reset infected
    @staticmethod
    def Reset(graph):
        for node, d in graph.nodes_iter(data=True): # Nodes
            d['infected'] = False

    # One cascade of the LT model in graph G starting from node a
    @staticmethod
    def Cascade(graph, seed, seedIdx):
        # Activate seed
        seed['infected'] = True
        seed['count'] += 1

        # Do cascade step while a node is still actuve
        nextChecks = LT.LTStep(graph, list(graph[seedIdx].keys()))
        while len(nextChecks) > 0:
            nextChecks = LT.LTStep(graph, nextChecks)


    # One step of an LT cascade in graph G
    @staticmethod
    def LTStep(graph, checks):
        nextChecks = []

        for n in checks: # Check if weight exceeds threshold this iteration
           if(sum([value['weight'] for key, value in graph[n].items() if (graph.node[key]['infected'] == True)]) > graph.node[n]['threshold']):
               graph.node[n]['infected'] = True
               graph.node[n]['count'] += 1

               nextChecks.extend([idx for idx in graph[n].keys() if idx not in nextChecks]) # Add the neighbors of the newly infected to check

           nextChecks = [idx for idx in nextChecks if graph.node[idx]['infected'] == False] # Filter next batch of nodes to check to only non-infected

        return nextChecks