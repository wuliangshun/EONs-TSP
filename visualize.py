# visualize.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_btsp_graph(best_path, U):
    G = nx.Graph()
    n = len(best_path)
    for i in range(n):
        G.add_node(i)
        j = (i + 1) % n
        G.add_edge(best_path[i], best_path[j], weight=np.round(U[best_path[i], best_path[j]], 3))

    pos = nx.circular_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("BTSP Solution: Channel Ordering")
    plt.show()
