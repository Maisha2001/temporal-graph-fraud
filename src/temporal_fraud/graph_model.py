import numpy as np
import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class TemporalGraphClassifier(nn.Module):
    def __init__(self, features, hidden=32, neighbors=8):
        super().__init__()
        if hidden < 2 or hidden % 2 or neighbors < 0:
            raise ValueError('Even hidden width and nonnegative neighbor budget required')
        self.features, self.hidden, self.neighbors = features, hidden, neighbors
        self.encode = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        if neighbors:
            self.role = nn.Embedding(2, hidden)
            self.conv = TransformerConv((hidden, hidden), hidden // 2, heads=2,
                                        beta=True, edge_dim=4, dropout=0.0)
        self.classify = nn.Sequential(nn.Linear(hidden * (3 if neighbors else 1), hidden),
                                      nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, rows, sample=None):
        query = self.encode(features[rows])
        if not self.neighbors:
            return self.classify(query).squeeze(-1)
        if sample is None:
            raise ValueError('Causal neighborhood required')
        history, ages, directions = sample
        history = torch.as_tensor(history.reshape(-1, self.neighbors), device=features.device, dtype=torch.long)
        keep = history >= 0
        roots = torch.arange(len(rows) * 2, device=features.device).repeat_interleave(self.neighbors)[keep.flatten()]
        messages = self.encode(features[history[keep]])
        age = torch.as_tensor(ages.reshape(-1, self.neighbors), device=features.device)[keep]
        direction = torch.as_tensor(directions.reshape(-1, self.neighbors), device=features.device)[keep]
        edge_features = torch.stack([torch.log1p(age) / 10, torch.sin(age * (2 * np.pi / 1440)),
                                     torch.cos(age * (2 * np.pi / 1440)), direction], dim=1)
        roles = self.role(torch.arange(2, device=features.device)).repeat(len(rows), 1)
        root_features = query.repeat_interleave(2, dim=0) + roles
        edge_index = torch.stack([torch.arange(len(roots), device=features.device), roots])
        context = self.conv((messages, root_features), edge_index, edge_features).relu()
        return self.classify(torch.cat([query, context.reshape(len(rows), -1)], dim=1)).squeeze(-1)


def predict(model, features, index, rows, batch_size=4096):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            sample = index.sample(batch, model.neighbors) if model.neighbors else None
            logits = model(features, torch.as_tensor(batch, device=features.device), sample)
            scores.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(scores)
