import itertools
import json as js
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np


# Open experiment file, create small files from the information file
def DivideFile(i, subdir, filename):
    with open(filename, 'r') as file:
        # Open file
        data = js.load(file)

        # Make directory for the file
        dirname = filename[:-4]
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Save snaps into individual files
        for snapID, snapVal in data['NetworkSnaps'].items():
            graph = nx.Graph()

            nodes = [(int(key.replace("'", '')), value) for key, value in snapVal['Nodes'].items()]
            edges = [(key.split('-')[0], key.split('-')[1], value) for key, value, in snapVal['Edges'].items()]
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)

            nx.write_gexf(graph, dirname + '/Snap ' + snapID + '-' + i + '.gexf')

        # Save growth values
        with open(dirname + '/Growth', 'w') as growthFile:
            js.dump(data['Result']['TrackedVal'], growthFile)

        # Save stats
        with open(dirname + '/Stats', 'w') as growthFile:
            js.dump(data['Result']['individualStats'], growthFile)


# Open .GEXF file, return array containing values of chosen attributes
def ExtractAttribute(graph, attName):
    try:
        return [d[attName] for n, d in graph.nodes(data=True)]
    except KeyError:
        return [d[attName] for u, v, d in graph.edges(data=True)]


# Plot the average histogram of polarization for all experiments inside the given folder path
def PlotAvg2dHistogram(path, title):
    # Collect data
    xVal = []
    yVal = []

    for exp in range(50):
        G = nx.read_gexf(os.path.join(path, str(exp + 1) + '-1-Snap .gexf'))
        xVal += [0] * G.order()
        yVal += [d['q'] for n1, d in G.nodes(data=True)]

        for j in range(100):
            G = nx.read_gexf(os.path.join(path, str(exp + 1) + '-' + str(((j + 1) * 100)) + '-Snap .gexf'))
            xVal += [((j + 1) * 100)] * G.order()
            yVal += [d['q'] for n1, d in G.nodes(data=True)]

        print(exp)

    # Plot 2d histogram

    xEdges = np.arange(0, 10200, 200)
    yEdges = np.arange(-1, 1.05, 0.05)

    H, xEdges, yEdges = np.histogram2d(yVal, xVal, bins=(yEdges, xEdges))
    # print(H)

    font = {'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.xlabel('# Cascade')
    plt.ylabel('Polarization')
    plt.imshow(H, interpolation='nearest', origin='lower', extent=[yEdges[0], yEdges[-1], xEdges[0], xEdges[-1]],
               aspect='auto', cmap='Blues', norm=colors.PowerNorm(0.60))
    plt.savefig(path + '-' + str(exp) + '.png')


# Plot transparent stacked 1D histograms
def PlotAvg1dHistograms(paths, beta):
    font = {'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(20, 10))

    count = 0
    for path in paths:
        single = []

        for exp in range(50):
            a = os.path.join(path, str(exp + 1) + '-10000-Snap .gexf')
            G = nx.read_gexf(os.path.join(path, str(exp + 1) + '-10000-Snap .gexf'))
            single += [d['q'] for n, d in G.nodes(data=True)]

        plt.hist(single, 50, range=(-1, 1), histtype='step', fill=False, ls='--', lw=3.0, label=str(beta[count]))
        count += 1

    plt.title('Averaged Distribution of q; t = 10000, ' + r'$\alpha = 0.50$, $\theta = 0.25$ varying $\beta$')

    plt.ylabel('# Nodes')
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / 50))
    plt.axes().yaxis.set_major_formatter(ticks_y)

    plt.xlabel('Polarization')

    plt.legend()
    plt.show()


# Calculate the Macy Polarization score of a given graph
def CalcPolarizationMacy(graph):
    qVals = [d['q'] for n, d in graph.nodes(data=True)]
    qComb = list(itertools.combinations(qVals, 2))
    return np.var(np.asarray([1 - abs(pair[0] - pair[1]) for pair in qComb]))


def GraphAvgMacyGrowth(paths, labels):
    font = {'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(20, 10))

    for path in paths:
        total = []
        data = []

        for exp in range(5):
            G = nx.read_gexf(os.path.join(path, str(exp + 1) + '-1-Snap .gexf'))
            data.append(CalcPolarizationMacy(G))

        total.append(data)

        for casc in range(100):
            data = []

            for exp2 in range(5):
                G = nx.read_gexf(os.path.join(path, str(exp2 + 1) + '-' + str((casc + 1) * 100) + '-Snap .gexf'))
                data.append(CalcPolarizationMacy(G))

            total.append(data)

        plt.plot(range(len(total)), [np.mean(np.asarray(a)) for a in total], linewidth=3, linestyle='dashed',
                 marker='s', markersize=6)

    plt.grid()
    plt.show()


# MAIN
########################

# alphas = ['0.00', '0.25']
# betas = ['1']

# for a in alphas:
#     for b in betas:
#         path = os.path.join(os.path.expanduser('~'), 'Documents', 'Research', 'Result', 'Newest', 'Model 2', 'Alpha ' + a + ' Beta ' + b)
#         PlotAvg2dHistogram(path, r'$\alpha=$' + a + r', $\beta=$' + b)


alphas = ['0']
betas = ['1']
ran = ['0.05', '0.15', '0.25']

for a in alphas:
    for b in betas:
        for r in ran:
            path = os.path.join(os.path.expanduser('~'), 'Documents', 'Research', 'Result', 'Newest', 'Model 3', 'A 0',
                                'Alpha ' + a + ' Beta ' + b + ' Range ' + r)
            PlotAvg2dHistogram(path, r'$\alpha=$' + a + r', $\beta=$' + b + ', range=' + r)


###############
# GraphAvgMacyGrowth(paths, [])

#############

# betas = ['0.01', '0.50', '1', '10', '1000']
# paths = []
#
# for b in betas:
#     paths.append(os.path.join(os.path.expanduser('~'), 'Documents', 'Research', 'Result', 'Newest', 'Model 3', 'A 0.50',
#                         'Alpha 0.50 Beta ' + b + ' Range 0.25'))
#
# PlotAvg1dHistograms(paths, beta=betas)

###############
# G = nx.read_gexf(os.path.join(path, '1-10000-Snap .gexf'))
# CalcPolarizationMacy(G)
# PlotAvg2dHistogram(path)
