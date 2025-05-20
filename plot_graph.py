import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
import os

# 1) Load the Cora dataset
dataset = Planetoid(root='path/to/your/data', name='Cora')
data = dataset[0]

# 2) Convert to an undirected NetworkX graph
G = to_networkx(data, to_undirected=True)

# 3) Compute PageRank centrality as an “importance” measure
pagerank = nx.pagerank(G, alpha=0.85, tol=1e-6)

# # pick the top-K by PageRank
# K = 300
# top_nodes = sorted(pr, key=pr.get, reverse=True)[:K]

# # induce the subgraph on just those nodes
# H = G.subgraph(top_nodes)

# # recompute positions on the smaller graph
# pos_H = nx.spring_layout(H, seed=42)

# plt.figure(figsize=(8,8))
# nodes = nx.draw_networkx_nodes(
#     H, pos_H,
#     node_color=[pr[n] for n in H.nodes()],
#     cmap=plt.cm.viridis,
#     node_size=50,
#     alpha=0.9,
# )
# nx.draw_networkx_edges(H, pos_H, alpha=0.4)
# plt.title(f"Top {K} Cora Nodes by PageRank")
# plt.axis('off')

# # Add colorbar legend
# plt.colorbar(nodes, label='PageRank Value')
# # Shrink and position colorbar in the corner
# plt.gcf().axes[1].set_box_aspect(10)  # Make colorbar thinner
# plt.gcf().axes[1].set_position([0.85, 0.3, 0.03, 0.4])  # Move to right side
# plt.savefig('cora_topK.png', dpi=300, bbox_inches='tight')
# plt.show()


# Calculate various centrality metrics
in_degree = dict(G.degree())  # Changed from in_degree to degree for undirected graph
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
katz = nx.katz_centrality(G, alpha=0.01)  # Reduced alpha value for better convergence

# Read the bit allocation data
bit_data = pd.read_csv('conv2_lin_fea_quant_bit_bit_mapping.csv')
bit_values = bit_data['bit_width'].str.strip('[]').astype(float).values

# Create subplots with colored bars based on bits
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Centrality Metrics in Cora Graph (Colored by Bit Allocation)')

# Function to create histogram with colored bars
def plot_colored_hist(values, bits, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate quantiles to set x-axis limits
    q1, q99 = np.quantile(values, [0.01, 0.99])
    
    n, bins, patches = ax.hist(values, bins=50, range=(q1, q99))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate average bit for each bar
    bin_bits = []
    for i in range(len(bins)-1):
        mask = (values >= bins[i]) & (values < bins[i+1])
        avg_bit = bits[mask].mean() if mask.any() else 0
        bin_bits.append(avg_bit)
    
    # Color bars based on average bits
    norm = plt.Normalize(1, 3)  # Fixed range from 0 to 4
    for patch, bit in zip(patches, bin_bits):
        patch.set_facecolor(plt.cm.viridis(norm(bit)))
    ax.set_title(title)
    
    # Add text showing percentage of data shown
    total = len(values)
    shown = sum((values >= q1) & (values <= q99))
    ax.text(0.95, 0.95, f'{shown/total*100:.1f}% of data', 
            transform=ax.transAxes, ha='right', va='top')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    plt.colorbar(sm, label='Average Bits')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create distributions directory if it doesn't exist
if not os.path.exists('distributions'):
    os.makedirs('distributions')

# Create individual plots
plot_colored_hist(list(in_degree.values()), bit_values, 'In-Degree Distribution', 'distributions/in_degree_dist.png')
plot_colored_hist(list(betweenness.values()), bit_values, 'Betweenness Distribution', 'distributions/betweenness_dist.png')
plot_colored_hist(list(closeness.values()), bit_values, 'Closeness Distribution', 'distributions/closeness_dist.png')
plot_colored_hist(list(eigenvector.values()), bit_values, 'Eigenvector Distribution', 'distributions/eigenvector_dist.png')
plot_colored_hist(list(pagerank.values()), bit_values, 'PageRank Distribution', 'distributions/pagerank_dist.png')
plot_colored_hist(list(katz.values()), bit_values, 'Katz Distribution', 'distributions/katz_dist.png')
