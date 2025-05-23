import inspect
from collections import OrderedDict
import networkx as nx
import csv


import torch
from torch.nn import Parameter, Module, ModuleDict
import torch.nn.functional as F
from torch_geometric.utils import (
    softmax,
    add_self_loops,
    remove_self_loops,
    add_remaining_self_loops,
    degree,
)
import torch_scatter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import numpy as np
import matplotlib.pyplot as plt


def scatter_(name, src, index, dim=0, dim_size=None):
    """Taken from an earlier version of PyG"""
    assert name in ["add", "mean", "min", "max"]

    op = getattr(torch_scatter, "scatter_{}".format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == "max":
        out[out < -10000] = 0
    elif name == "min":
        out[out > 10000] = 0

    return out

def compute_global_centralities(data):
    # 1) Build an undirected NetworkX graph from edge_index
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edges)

    # 2) Compute centrality measures
    betweenness   = nx.betweenness_centrality(G)        # too sparse - no pattern
    print("Betweenness Centrality Computed")
    closeness     = nx.closeness_centrality(G)
    print("Closeness Centrality Computed")
    eigenvector   = nx.eigenvector_centrality(G)
    print("Eigenvector Centrality Computed")
    pagerank      = nx.pagerank(G)
    print("Pagerank Computed")
    katz          = nx.katz_centrality(G, alpha=0.005)  # choose alpha<1/λ_max
    print("Katz Centrality Computed")

    # 3) Pack into tensors aligned with node indices
    n = data.num_nodes
    bc = torch.tensor([betweenness[i] for i in range(n)])
    cc = torch.tensor([closeness[i] for i in range(n)])
    ec = torch.tensor([eigenvector[i] for i in range(n)])
    pr = torch.tensor([pagerank[i] for i in range(n)])
    kc = torch.tensor([katz[i] for i in range(n)])
    
    return bc, cc, ec, pr, kc

def analysis_bit(data, state_dict,all_positive=True,name='plat'):
    mean_all = []
    if(name=='ogbn-arxiv'):
        adj_t = data.adj_t
        adj_t = adj_t.fill_value(1.,)
        deg = sparsesum(adj_t, dim=1)
    else:
        edge_index = data.edge_index
        row,col = edge_index
        deg = degree(col,data.x.size(0))
    # bc, cc, ec, pr, kc = compute_global_centralities(data)
    for key in state_dict.keys():
        if ('quant' in key and 'bit' in key and 'fea' in key):
            if(all_positive):
                bit=state_dict[key].abs().round()-1
            else:
                bit=state_dict[key].abs().round()-1
            
            # # —– Save mapping for this layer —–
            # layer_name = key.replace('.', '_').replace('/', '_')
            # fname = f"{layer_name}_bit_mapping.csv"
            # with open(fname, 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(['index', 'bit_width'])
            #     for idx, b in enumerate(bit.tolist()):
            #         writer.writerow([idx, b])
            # print(f"  → Saved bit mapping to `{fname}`")

            print(key+'\n')
            mean_all.append(bit.mean())
            print('The average bits of current layer:',bit.mean())
            print("0bit:{}".format((bit==0).sum()))
            print("1bit:{}".format((bit==1).sum()))
            print("2bit:{}".format((bit==2).sum()))
            print("3bit:{}".format((bit==3).sum()))
            print("4bit:{}".format((bit==4).sum()))
            print("5bit:{}".format((bit==5).sum()))
            print("6bit:{}".format((bit==6).sum()))
            print("7bit:{}".format((bit==7).sum()))
            print("8bit:{}".format((bit==8).sum()))
            print("9bit:{}".format((bit==9).sum()))
            print('\n')
            print('The average degree of the nodes using corresponding bitwidth:')
            index_1_bit = torch.where(bit==1)[0]
            index_2_bit = torch.where(bit==2)[0]
            index_3_bit = torch.where(bit==3)[0]
            index_4_bit = torch.where(bit==4)[0]
            index_5_bit = torch.where(bit==5)[0]
            index_6_bit = torch.where(bit==6)[0]
            index_7_bit = torch.where(bit==7)[0]
            index_8_bit = torch.where(bit==8)[0]
            print('1bit_deg_mean:',deg[index_1_bit].mean())
            print('2bit_deg_mean:',deg[index_2_bit].mean())
            print('3bit_deg_mean:',deg[index_3_bit].mean())
            print('4bit_deg_mean:',deg[index_4_bit].mean())
            print('5bit_deg_mean:',deg[index_5_bit].mean())
            print('6bit_deg_mean:',deg[index_6_bit].mean())
            print('7bit_deg_mean:',deg[index_7_bit].mean())
            print('8bit_deg_mean:',deg[index_8_bit].mean())
            print('\n')

            # print('1bit_betweenness_mean:',bc[index_1_bit].mean())
            # # print(bc[index_1_bit])
            # print('2bit_betweenness_mean:',bc[index_2_bit].mean())
            # # print(bc[index_2_bit])
            # print('3bit_betweenness_mean:',bc[index_3_bit].mean())
            # # print(bc[index_3_bit])
            # print('4bit_betweenness_mean:',bc[index_4_bit].mean())
            # # print(bc[index_4_bit])
            # print('5bit_betweenness_mean:',bc[index_5_bit].mean())
            # # print(bc[index_5_bit])
            # print('6bit_betweenness_mean:',bc[index_6_bit].mean())
            # # print(bc[index_6_bit])
            # print('7bit_betweenness_mean:',bc[index_7_bit].mean())
            # # print(bc[index_7_bit])
            # print('8bit_betweenness_mean:',bc[index_8_bit].mean())
            # # print(bc[index_8_bit])
            # print('\n')

            # print('1bit_closeness_mean:',cc[index_1_bit].mean())
            # # print(cc[index_1_bit])
            # print('2bit_closeness_mean:',cc[index_2_bit].mean())
            # # print(cc[index_2_bit])
            # print('3bit_closeness_mean:',cc[index_3_bit].mean())
            # # print(cc[index_3_bit])
            # print('4bit_closeness_mean:',cc[index_4_bit].mean())
            # # print(cc[index_4_bit])
            # print('5bit_closeness_mean:',cc[index_5_bit].mean())
            # # print(cc[index_5_bit])
            # print('6bit_closeness_mean:',cc[index_6_bit].mean())
            # # print(cc[index_6_bit])
            # print('7bit_closeness_mean:',cc[index_7_bit].mean())
            # # print(cc[index_7_bit])
            # print('8bit_closeness_mean:',cc[index_8_bit].mean())
            # # print(cc[index_8_bit])
            # print('\n')

            # print('1bit_eigenvector_mean:',ec[index_1_bit].mean())
            # print('2bit_eigenvector_mean:',ec[index_2_bit].mean())
            # print('3bit_eigenvector_mean:',ec[index_3_bit].mean())
            # print('4bit_eigenvector_mean:',ec[index_4_bit].mean())
            # print('5bit_eigenvector_mean:',ec[index_5_bit].mean())
            # print('6bit_eigenvector_mean:',ec[index_6_bit].mean())
            # print('7bit_eigenvector_mean:',ec[index_7_bit].mean())
            # print('8bit_eigenvector_mean:',ec[index_8_bit].mean())
            # print('\n')

            # print('1bit_pagerank_mean:',pr[index_1_bit].mean())
            # print('2bit_pagerank_mean:',pr[index_2_bit].mean())
            # print('3bit_pagerank_mean:',pr[index_3_bit].mean())
            # print('4bit_pagerank_mean:',pr[index_4_bit].mean())
            # print('5bit_pagerank_mean:',pr[index_5_bit].mean())
            # print('6bit_pagerank_mean:',pr[index_6_bit].mean())
            # print('7bit_pagerank_mean:',pr[index_7_bit].mean())
            # print('8bit_pagerank_mean:',pr[index_8_bit].mean())
            # print('\n')

            # print('1bit_katz_mean:',kc[index_1_bit].mean())
            # print('2bit_katz_mean:',kc[index_2_bit].mean())
            # print('3bit_katz_mean:',kc[index_3_bit].mean())
            # print('4bit_katz_mean:',kc[index_4_bit].mean())
            # print('5bit_katz_mean:',kc[index_5_bit].mean())
            # print('6bit_katz_mean:',kc[index_6_bit].mean())
            # print('7bit_katz_mean:',kc[index_7_bit].mean())
            # print('8bit_katz_mean:',kc[index_8_bit].mean())
            # print('\n')
    print('The average bits: ',sum(mean_all)/len(mean_all))
    print('Finish')

msg_special_args = set(
    [
        "edge_index",
        "edge_index_i",
        "edge_index_j",
        "size",
        "size_i",
        "size_j",
    ]
)

aggr_special_args = set(
    [
        "index",
        "dim_size",
    ]
)

update_special_args = set([])
