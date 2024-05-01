# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim

class SSTKGPreprocess(nn.Module):
    def __init__(self, num_entities):
        super(SSTKGPreprocess, self).__init__()
        self.p = nn.Parameter(torch.randn(1))
        self.I = nn.Parameter(torch.randn(num_entities))

    def forward(self, records, e0_index):
        center_record = records[e0_index]
        other_records = torch.cat((records[:e0_index], records[e0_index+1:]))
        return self.p * center_record - torch.matmul(self.I, other_records)

def construction_model(records, e0_index, num_entities, epochs=1000, lr=0.01):
    model = SSTKGPreprocess(num_entities)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    records_tensor = torch.tensor(records, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(records_tensor, e0_index)
        loss = criterion(outputs, torch.zeros_like(outputs))
        loss.backward()
        optimizer.step()
    
    return model.p.item(), model.I.detach().numpy()
