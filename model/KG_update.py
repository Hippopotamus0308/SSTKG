import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from KG_training import SSTKG, train, predict_records

def load_kg(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_kg(state, path):
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def visualize_kg(kg, time_step):
    G = nx.DiGraph()
    influence_matrix_at_time = kg['influence_matrices'][time_step]
    for i, influences in enumerate(influence_matrix_at_time):
        for j, influence in enumerate(influences):
            if influence > 0.05: 
                G.add_edge(i, j, weight=influence)
    
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()

def update_kg(data_loader, model, n_epochInf, n_epochEmb, optimizer):
    train(model, data_loader, n_epochInf, n_epochEmb, optimizer)
    kg_state = {
        'out_embeddings': [emb.data.numpy() for emb in model.out_embeddings],
        'influence_matrices': [inf.data.numpy() for inf in model.influence_matrices]
    }
    return kg_state

def build_graph_from_kg(kg, time_step):
    G = nx.DiGraph()
    influence_matrix = kg['influence_matrices'][time_step]
    for i, influences in enumerate(influence_matrix):
        for j, influence in enumerate(influences):
            if influence > 0.05: 
                G.add_edge(i, j, weight=influence)
    return G

def find_path(kg, time_step, source, target, max_hops=None):
    G = build_graph_from_kg(kg, time_step)
    if max_hops:
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
    else:
        paths = list(nx.all_simple_paths(G, source=source, target=target))
    if paths:
        print(f"Paths from {source} to {target} at time step {time_step}: {paths}")
    else:
        print(f"No path found from {source} to {target} at time step {time_step}.")

def check_connection(kg, time_step, source, target):
    G = build_graph_from_kg(kg, time_step)
    connected = nx.has_path(G, source, target)
    print(f"Entities {source} and {target} are {'connected' if connected else 'not connected'} at time step {time_step}.")

def get_weight(kg, time_step, source, target):
    G = build_graph_from_kg(kg, time_step)
    try:
        weight = G[source][target]['weight']
        print(f"Weight from {source} to {target} at time step {time_step} is {weight}.")
    except KeyError:
        print(f"No direct edge from {source} to {target} at time step {time_step}.")

def get_top_influencers(kg, time_step, entity_id, top_n=5):
    G = build_graph_from_kg(kg, time_step)
    if entity_id in G:
        influencers = sorted(G.in_edges(entity_id, data='weight'), key=lambda x: x[2], reverse=True)
        print(f"Top {top_n} influencers for entity {entity_id} at time step {time_step}:")
        for influencer in influencers[:top_n]:
            print(f"Entity {influencer[0]} with influence weight: {influencer[2]}")
    else:
        print(f"No influencers found for entity {entity_id} at time step {time_step}.")

def analyze_influence_change(kg, entity_id):
    time_steps = len(kg['influence_matrices'])
    influence_changes = []
    for time_step in range(time_steps):
        G = build_graph_from_kg(kg, time_step)
        total_influence = sum([weight for _, _, weight in G.edges(entity_id, data='weight')])
        influence_changes.append(total_influence)
        print(f"Total influence of entity {entity_id} at time step {time_step}: {total_influence}")
    
    plt.plot(range(time_steps), influence_changes)
    plt.xlabel("Time Step")
    plt.ylabel("Total Influence")
    plt.title(f"Influence Over Time for Entity {entity_id}")
    plt.show()

def compare_embedding_distance(kg, entity_id1, entity_id2):
    time_steps = len(kg['out_embeddings'])
    distances = []
    for time_step in range(time_steps):
        emb1 = kg['out_embeddings'][time_step][entity_id1]
        emb2 = kg['out_embeddings'][time_step][entity_id2]
        distance = torch.norm(emb1 - emb2, p=2).item()
        distances.append(distance)
        print(f"Distance between entity {entity_id1} and {entity_id2} at time step {time_step}: {distance}")
    
    plt.plot(range(time_steps), distances)
    plt.xlabel("Time Step")
    plt.ylabel("Euclidean Distance")
    plt.title(f"Embedding Distance Over Time for Entities {entity_id1} and {entity_id2}")
    plt.show()

def analyze_network_density(kg):
    time_steps = len(kg['influence_matrices'])
    densities = []
    for time_step in range(time_steps):
        G = build_graph_from_kg(kg, time_step)
        density = nx.density(G)
        densities.append(density)
        print(f"Network density at time step {time_step}: {density}")
    
    plt.plot(range(time_steps), densities)
    plt.xlabel("Time Step")
    plt.ylabel("Density")
    plt.title("Network Density Over Time")
    plt.show()

def predict_entity_record(model, entity_id, time_step, related_entities):
    """
    Predict and display the records for a specific entity at a specific time step.
    """
    predicted_record = predict_records(model, entity_id, time_step, related_entities)
    print(f"Predicted record for entity {entity_id} at time step {time_step}: {predicted_record}")

def compare_prediction_to_actual(model, entity_id, actual_record, time_step, related_entities):
    """
    Compare the predicted records to actual records for a specific entity.
    """
    predicted_record = predict_records(model, entity_id, time_step, related_entities)
    print(f"Predicted record: {predicted_record}")
    print(f"Actual record: {actual_record}")
    mse = torch.mean((predicted_record - actual_record) ** 2).item()
    print(f"Mean squared error between predicted and actual record: {mse}")

def evaluate_predictions(model, entity_ids, actual_records, time_steps, related_entities_list):
    """
    Evaluate the predictions for multiple entities and time steps.
    """
    mse_total = 0
    for entity_id, actual_record, time_step, related_entities in zip(entity_ids, actual_records, time_steps, related_entities_list):
        predicted_record = predict_records(model, entity_id, time_step, related_entities)
        mse = torch.mean((predicted_record - actual_record) ** 2).item()
        mse_total += mse
        print(f"Entity {entity_id} at time step {time_step} - MSE: {mse}")
    average_mse = mse_total / len(entity_ids)
    print(f"Average MSE for all predictions: {average_mse}")