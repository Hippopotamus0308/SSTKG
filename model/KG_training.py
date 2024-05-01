import torch
import torch.nn as nn
import torch.optim as optim
import pickle

class SSTKG(nn.Module):
    def __init__(self, num_entities, embedding_dim, pretrained_influence, category_embeddings, overall_records, initial_temporal_records):
        super(SSTKG, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.influence_matrix = nn.Parameter(pretrained_influence)

        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), 
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
        self.static_embeddings = nn.Parameter(self.init_static_embeddings(category_embeddings, overall_records))

        self.out_embeddings = nn.ParameterList([nn.Parameter(self.init_dynamic_embeddings(self.static_embeddings, temporal_records)) for temporal_records in initial_temporal_records])

    def init_static_embeddings(self, category_embeddings, overall_records):
        overall_records_processed = torch.sigmoid(overall_records)
        combined_input = torch.cat([category_embeddings, overall_records_processed], dim=1)
        return self.embedding_processor(combined_input)
    
    def init_dynamic_embeddings(self, static_embeddings, initial_temporal_records):
        combined_input = torch.cat([static_embeddings, initial_temporal_records], dim=1)
        return self.psi(combined_input)

    def forward(self, e_0, temporal_records):
        static_emb = self.static_embeddings[e_0]
        out_emb = out_emb = self.psi(static_emb, temporal_records)
        in_emb = torch.matmul(self.influence_matrix[e_0], self.out_embeddings)
        return static_emb, out_emb, in_emb
    
    @staticmethod
    def psi(combined_input):
        embedding_dim = combined_input.shape[1] // 2 
        linear = nn.Linear(combined_input.shape[1], embedding_dim)
        activation = nn.Tanh()
        return activation(linear(combined_input))
    
    def score1(self, e_0, p_e_0, temporal_records):
        _, out_emb, in_emb = self(e_0, temporal_records)
        return torch.norm(p_e_0 * out_emb - in_emb, p=2)**2

    def score2(self, e_0, R, temporal_records):
        _, out_emb, _ = self(e_0, temporal_records)
        p_e_0 = self.influence_matrix[e_0, e_0]
        sum_influence_outs = torch.sum(R * self.out_embeddings, axis=0)
        return torch.norm(p_e_0 * out_emb - sum_influence_outs, p=2)**2

    def loss1(self, score1_batch):
        losses = -torch.sum(torch.log(self.sigmoid(score1_batch - score1_batch.unsqueeze(1))), dim=1)
        return torch.mean(losses)

    def loss2(self, score2_batch):
        losses = -torch.sum(torch.log(self.sigmoid(score2_batch - score2_batch.unsqueeze(1))), dim=1)
        return torch.mean(losses)

    def calculate_static_embedding(self, overall_records):
        return self.phi(overall_records)

    def save_state(self, filename):
        state = {
            'out_embeddings': [emb.data for emb in self.out_embeddings],
            'influence_matrices': [inf.data for inf in self.influence_matrices]
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for i, emb in enumerate(state['out_embeddings']):
            self.out_embeddings[i] = nn.Parameter(emb)
        for i, inf in enumerate(state['influence_matrices']):
            self.influence_matrices[i] = nn.Parameter(inf)


def train(model, data_loader, n_epochInf, n_epochEmb, optimizer):
    # Phase 1: Training embeddings for each time step
    for epoch in range(n_epochInf):
        for time_step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            loss_inf = 0
            for e_0, R, temporal_records in batch:
                p_e_0 = model.influence_matrices[time_step][e_0, e_0]  # Get self influence for the current time step
                _, out_emb, in_emb = model(e_0, time_step)
                score1_val = torch.norm(p_e_0 * out_emb - in_emb, p=2)**2
                loss1_val = model.loss1(score1_val.unsqueeze(0))  # Unsqueezing since loss1 expects batched scores

                loss_inf += loss1_val
            loss_inf.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Time Step {time_step}, Phase 1, Loss: {loss_inf.item()}')

    # Phase 2: Training influence matrices for each time step
    for epoch in range(n_epochEmb):
        for time_step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            loss_emb = 0
            for e_0, R, temporal_records in batch:
                _, out_emb, _ = model(e_0, time_step)
                score2_val = torch.norm(p_e_0 * out_emb - torch.matmul(R, model.out_embeddings[time_step]), p=2)**2
                loss2_val = model.loss2(score2_val.unsqueeze(0))  # Unsqueezing since loss2 expects batched scores

                loss_emb += loss2_val
            loss_emb.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Time Step {time_step}, Phase 2, Loss: {loss_emb.item()}')


def predict_records(sstkg_model, entity_id, time_step, related_entities):
    """
    Predict the records for a given entity based on its related entities' embeddings.

    Args:
    sstkg_model (SSTKG): An instance of the SSTKG model.
    entity_id (int): The target entity ID for which to predict the records.
    time_step (int): The time step at which the prediction is made.
    related_entities (list of int): A list of entity IDs that are related to the target entity.

    Returns:
    torch.Tensor: The predicted records for the target entity at the given time step.
    """
    related_static_embs = sstkg_model.static_embeddings[related_entities]
    related_out_embs = [sstkg_model.out_embeddings[time_step][rel_ent] for rel_ent in related_entities]

    influences = [sstkg_model.influence_matrices[time_step][entity_id, rel_ent] for rel_ent in related_entities]
    weighted_sums = torch.sum(torch.stack([inf * out_emb for inf, out_emb in zip(influences, related_out_embs)], dim=0), dim=0)

    in_embedding = torch.sum(weighted_sums + related_static_embs, dim=0)

    predicted_records = decoder(in_embedding)

    return predicted_records