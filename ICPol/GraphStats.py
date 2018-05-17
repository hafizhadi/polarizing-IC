import math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# Return statistics of an array of sample
def CalcStats(samples):
    return dict(zip(['mean', 'max', 'min', 'sdeviation'],
                    [float(sp.mean(samples)), float(sp.nanmax(samples)), float(sp.nanmin(samples)),
                     float(sp.std(samples))]))


# Analyze an initialized graph, return a dictionary
def AnalyzeGraph(graph):
    analysisRes = {}  # Result

    # Create lists of attributes
    q = []
    rejCount = []
    accCount = []
    propCount = []
    wEdgeHom = []
    histQ = []

    flowCount = []
    c = []

    # Collect node data
    for n, d in graph.nodes(data=True):
        q.append(d["q"])
        rejCount.append(d['rejCount'])
        accCount.append(d['accCount'])
        propCount.append(d['propCount'])
        wEdgeHom.append(d['wEdgeHom'])
        histQ.append(d['histQ'])

        # Add the information to the graph
        graph.node[n]['degree'] = graph.degree(n)

    # Calculate some attributes for the edges
    for a, b, d in graph.edges(data=True):
        flowCount.append(d['flowCount'])
        c.append(d['c'])

        # Calculate stats of each list
    nKeys = ['q', 'rejCount', 'accCount', 'propCount', 'wEdgeHom', 'histQ']
    nVals = [q, rejCount, accCount, propCount, wEdgeHom, histQ]

    eKeys = ['flowCount', 'c']
    eVals = [flowCount, c]

    analysisRes["individualStats"] = {}
    analysisRes["individualStats"]["nodes"] = dict(zip(nKeys, [CalcStats(v) for v in nVals]))
    analysisRes["individualStats"]["edges"] = dict(zip(eKeys, [CalcStats(v) for v in eVals]))

    return analysisRes


# Take a networkX graph object and do histogram visualizations
def VisualizeDistribution(graph, nodePlots, edgePlots, info, filename):
    plt.figure(figsize=(10, 10))
    names = nodePlots + edgePlots

    # Prepare data
    plotData = []
    for name in nodePlots:
        if (name == 'Degree'):
            plotData.append(np.array([graph.degree(x) for x in graph.nodes()]))
        else:
            plotData.append(np.array([data[name] for node, data in graph.nodes(data=True)]))

    for name in edgePlots:
        plotData.append(np.array([data[name] for a, b, data in graph.edges(data=True)]))

    # Calculate dimension of plot
    amount = len(plotData) + int(info != "")
    row = round(math.sqrt(amount) + 0.5)
    column = round((amount / row))

    # Plot using iteration
    i = 1
    for data in plotData:

        plt.subplot(row, column, i)

        if (names[i - 1] == 'q'):
            plt.xlim((-1, 1))
            plt.title("Polarization")
            plt.ylabel("# Nodes")
        elif (names[i - 1] == 'c'):
            plt.xlim((0, 1))
            plt.title("Connection Strength")
            plt.ylabel("# Edges")
        else:
            plt.title(names[i - 1])
            plt.ylabel("Count")

        plt.hist(data, 100)
        plt.axvline(data.mean(), color='r', linewidth=1, linestyle='dashed')
        plt.annotate("mean= " + str(data.mean()) + "\nsize=" + str(len(data)), xy=(data.mean(), 0.8),
                     xycoords=('data', 'axes fraction'),
                     xytext=(10, 0), textcoords='offset points')

        i += 1

    # Write Experiment Data
    plt.annotate(info, xy=(1, 0), xycoords=('axes fraction'), xytext=(50, -20), textcoords='offset points')

    plt.savefig(filename, dpi='figure')
    plt.close()

    return


# Take a dictionary of list of values over time and plot
def VisualizeGrowth(data, info, filename):
    plt.figure(figsize=(16, 16))

    # Calculate dimension of plot
    amount = len(data.items())
    row = round(math.sqrt(amount) + 0.5)
    column = round((amount / row) + 0.5)

    # Plot using iteration
    i = 1
    for key, value in data.items():
        plt.subplot(row, column, i)
        plt.title(key)
        plt.xlabel('#Cascade')

        plt.plot(range(len(value)), value)
        i += 1

    plt.savefig(filename, dpi='figure')
    plt.close()

    return