import math

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

COLOR = ['deeppink', 'crimson', 'darkorange', 'olive', 'green', 'indigo', 'gold', 'teal', 'dodgerblue', 'maroon',
         'dimgray']

# Return statistics of an array of sample
def DistStats(samples):
    return dict(zip(['mean', 'max', 'min', 'sdeviation'],
                    [float(sp.mean(samples)), float(sp.nanmax(samples)), float(sp.nanmin(samples)),
                     float(sp.std(samples))]))


# Analyze an initialized graph, return a dictionary
def AnalyzeAttributes(graph):
    # Create lists of attributes
    nodeAttributes = {key: [] for key, value in graph.nodes[list(graph.nodes())[0]].items()}
    edgeAttributes = {key: [] for key, value in graph.edges[list(graph.edges())[0]].items()}

    # Collect node data
    for n, d in graph.nodes(data=True):
        for key, value in d.items():
            nodeAttributes[key].append(value)

    # Collect edge data
    for a, b, d in graph.edges(data=True):
        for key, value in d.items():
            edgeAttributes[key].append(value)

    # Calculate stats of each list
    analysisRes = {}
    analysisRes["nodes"] = {key: DistStats(value) for key, value in nodeAttributes.items()}
    analysisRes["edges"] = {key: DistStats(value) for key, value in edgeAttributes.items()}

    return analysisRes


# Take a networkX graph object and get a list containing all the values for selected attributes
def GetValues(graph, nodeAtt, edgeAtt):
    attrValues = []
    for name in nodeAtt:
        if (name == 'Degree'):
            attrValues.append((name, np.asarray([graph.degree(x) for x in graph.nodes()])))
        else:
            attrValues.append((name, np.asarray([data[name] for node, data in graph.nodes(data=True)])))

    for name in edgeAtt:
        attrValues.append((name, np.asarray([data[name] for a, b, data in graph.edges(data=True)])))

    return attrValues


# Take a list of (name, values) dictionary and make histograms
def VisHistogram(values, filename, nExp=1):
    font = {'size': 25}
    plt.rc('font', **font)

    plt.figure(figsize=(30, 30))

    # Calculate dimension of plot
    amount = len(values)
    row = round(math.sqrt(amount) + 0.5)
    column = round((amount / row) + 0.5)

    # Plot using iteration
    i = 1
    for name, value in values:

        plt.subplot(row, column, i)

        if (name == 'q'):
            plt.xlim((-1, 1))
            plt.title("Polarization", y=1.05)
            plt.ylabel("# Nodes")
        elif (name == 'c'):
            plt.xlim((0, 1))
            plt.title("Connection Strength", y=1.05)
            plt.ylabel("# Edges")
        else:
            plt.title(name, y=1.05)
            plt.ylabel("Count")

        plt.hist(value, 50, color=COLOR[i])
        plt.grid()

        locs, labels = plt.yticks()
        newLocs = [str(x / nExp) for x in locs]
        plt.yticks(locs, newLocs)

        plt.axvline(value.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(round(value.mean(), 3)) + "\nsize=" + str(len(value) / nExp),
                     xy=(value.mean(), 0.8),
                     xycoords=('data', 'axes fraction'),
                     xytext=(10, 0), textcoords='offset points')

        i += 1

    plt.tight_layout()
    plt.savefig(filename, dpi='figure')
    plt.close()

    return


# Take a list of (name, values) dictionary and make histograms
def Vis2(histData, filename):
    font = {'size': 25}
    plt.rc('font', **font)

    plt.figure(figsize=(30, 30))

    # Calculate dimension of plot
    amount = len(histData)
    row = round(math.sqrt(amount) + 0.5)
    column = round((amount / row) + 0.5)

    # Plot using iteration
    i = 1
    for name, value in histData:

        plt.subplot(row, column, i)

        if (name == 'q'):
            plt.xlim((-1, 1))
            plt.title("Polarization", y=1.05)
            plt.ylabel("# Nodes")
        elif (name == 'c'):
            plt.xlim((0, 1))
            plt.title("Connection Strength", y=1.05)
            plt.ylabel("# Edges")
        else:
            plt.title(name, y=1.05)
            plt.ylabel("Count")

        width = 0.7 * (value[1][1] - value[1][0])
        center = (value[1][:-1] + value[1][1:]) / 2
        plt.bar(center, value[0], align='center', width=width, color=COLOR[i])
        plt.grid()

        i += 1

    plt.tight_layout()
    plt.savefig(filename, dpi='figure')
    plt.close()

    return

# Take a dictionary of list of values over time and plot
def VisGrowth(data, info, filename):
    font = {'size': 25}
    plt.rc('font', **font)
    plt.figure(figsize=(30, 30))

    # Calculate dimension of plot
    amount = len(data.items())
    row = round(math.sqrt(amount) + 0.5)
    column = round((amount / row) + 0.5)

    # Plot using iteration
    i = 1
    for key, value in data.items():
        plt.subplot(row, column, i)
        plt.title(key, y=1.05)
        plt.xlabel('#Cascade')

        plt.plot(range(len(value)), value, linewidth=3, linestyle='dashed', marker='s', markersize=6, color=COLOR[i])
        plt.grid()
        i += 1

    plt.tight_layout()
    plt.savefig(filename, dpi='figure')
    plt.close()

    return