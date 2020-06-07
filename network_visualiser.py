# Take incidence formats from tien mai code - (start_node, end_node, 1)  - sparse matrix
import os
from os.path import join
# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network
file = "Input" # big data from classical v2
folder = join("Datasets", file)
rows_to_keep = None  # currently am keeping all rows, but are hacking row length 0 to row length 1
INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
file_incidence = os.path.join(folder, INCIDENCE)
file_dist = os.path.join(folder, TRAVEL_TIME)

import numpy as np
import networkx as nx

# import pandas as pd
# df =pd.read_csv(file, sep=" ", header=None, names=["O", "D", "incidence"])
# print(df.head(),)
o_node, d_node = np.loadtxt(file_incidence, dtype='int', usecols=(0, 1), unpack=True)

G = nx.Graph()
G.add_nodes_from(o_node)
G.add_nodes_from(d_node)

# final 3 rows have distance 0- this is paths to the final node and they have zero cost?
# incident_edge_data = np.loadtxt(file_incidence, usecols=(0, 1))
# Input Dataset has inconsistent incidence -> don't use incidence
incident_edge_data = np.loadtxt(file_dist, usecols=(0, 1), max_rows=rows_to_keep)
G.add_edges_from(incident_edge_data)
dist_data = np.loadtxt(file_dist, max_rows=rows_to_keep)

# get minnonzero entry # note this is not stable (a view)
nonzero_min = np.ma.masked_equal(dist_data[:, 2], 0.0, copy=False).min()
# replace zeros with smallest length scale
dist_data[:,2] = np.maximum(dist_data[:,2], nonzero_min)
# normalise for nicer numbers
dist_data[:,2] = np.divide(dist_data[:,2], nonzero_min)
# note non_zero_dists is a mask, so this is a view and changes after normalise


dist_data[:,2] = np.maximum(dist_data[:,2],1) # replace 0 lengths with 1
print(dist_data[:, 0:2])
print(G.nodes)


for s,f, travel_time in dist_data:
    # print(s,f)
    G[int(s)][int(f)]['travel_time'] = travel_time
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges", )

import matplotlib.pyplot as plt

# nx.drawing.draw_networkx(G, pos=nx.spectral_layout(G,)) # unweighted - for structure
# plt.savefig(join("figures", f"{file} - network_structure.pdf"))
# plt.figure()
# note spectral weight is inverse weight?
layout = nx.drawing.kamada_kawai_layout(G, weight='travel_time')
nx.drawing.draw_networkx(G, pos=layout) # unweighted - for
# structure
plt.savefig(join("figures", f"{file} - network_dists - no dists.pdf"))
nx.drawing.draw_networkx_edge_labels(G, pos=layout,
                                     edge_labels={(i,j):round(k,1) for (i,j,k) in dist_data})
# networkx.drawing.draw_spectral(G)
# networkx.drawing.draw_spring(G)
# networkx.drawing.draw_planar(G) #no overlapping edges
from pprint import pprint
pprint({(i,j):round(k,1) for (i,j,k) in dist_data})
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(join("figures", f"{file} - network_dists.pdf"))
plt.show()
