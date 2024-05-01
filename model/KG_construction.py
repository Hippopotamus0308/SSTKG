# kg_construction.py
import numpy as np
from KG_preprocess import construction_model
import networkx as nx
import pickle
import matplotlib.pyplot as plt

def construct_kg(num_entities, records):
    G = nx.DiGraph()
    for e0 in range(num_entities):
        p_e0, I_values = train_model(records, e0, num_entities)
        for i, I_value in enumerate(I_values):
            if i < e0:
                target_node = i
            else:
                target_node = i + 1
            G.add_edge(e0, target_node, weight=I_value)
    
    with open('knowledge_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    return G

def show_graph(G):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def find_path(G, source, target, max_hops=None):
    if max_hops:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
    else:
        paths = list(nx.all_simple_paths(G, source=source, target=target))
    if paths:
        print(f"Paths from {source} to {target}: {paths}")
    else:
        print(f"No path found from {source} to {target}.")

def check_connection(G, source, target):
    connected = nx.has_path(G, source, target)
    print(f"Entities {source} and {target} are {'connected' if connected else 'not connected'}.")

def get_weight(G, source, target):
    try:
        weight = G[source][target]['weight']
        print(f"Weight from {source} to {target} is {weight}.")
    except KeyError:
        print(f"No direct edge from {source} to {target}.")

def save_graph(G, filename):
    nx.write_gpickle(G, filename)
    print(f"Graph saved as {filename}.")

def load_graph(filename):
    return nx.read_gpickle(filename)
