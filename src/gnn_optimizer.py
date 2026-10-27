"""Experimental message-passing rewrite ranker using only PyTorch.

Scores legal rewrites, never arbitrary gate deletion masks. Supervised targets
are bounded greedy-lookahead rewards, not a claim of optimality or novelty.
"""
import math
from pathlib import Path

import torch
from torch import nn

from .graph_optimizer import dependency_edges, rewrite_gain

GATES = ('x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'cx', 'cz',
         'swap', 'rx', 'ry', 'rz', 'id', 'barrier')
FEATURES = len(GATES) + 3
SCHEMA = 1


def graph_tensors(circuit):
    features = torch.zeros((len(circuit.data), FEATURES))
    for i, item in enumerate(circuit.data):
        features[i, GATES.index(item.operation.name)] = 1
        angle = float(item.operation.params[0]) if item.operation.params else 0
        features[i, -3:] = torch.tensor([math.sin(angle), math.cos(angle),
                                        len(item.qubits) / 2])
    edges = dependency_edges(circuit)
    indices = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).reshape(-1, 2)
    # Preserve the gate operand roles on both sides of a dependency edge.
    ports = torch.tensor([[e[2], e[3]] for e in edges], dtype=torch.float32).reshape(-1, 2)
    return features, indices, ports


class RewriteGNN(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.input = nn.Linear(FEATURES, hidden)
        self.forward_messages = nn.ModuleList(nn.Linear(hidden + 2, hidden) for _ in range(layers))
        self.backward_messages = nn.ModuleList(nn.Linear(hidden + 2, hidden) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Linear(hidden * 3, hidden) for _ in range(layers))
        self.score = nn.Sequential(nn.Linear(hidden * 3 + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, circuit, candidates):
        x, edges, ports = graph_tensors(circuit)
        h = torch.relu(self.input(x))
        source, dest = edges[:, 0], edges[:, 1]
        for fwd, bwd, update in zip(self.forward_messages, self.backward_messages, self.updates):
            incoming, outgoing = torch.zeros_like(h), torch.zeros_like(h)
            incoming.index_add_(0, dest, torch.relu(fwd(torch.cat([h[source], ports], dim=1))))
            outgoing.index_add_(0, source, torch.relu(bwd(torch.cat([h[dest], ports], dim=1))))
            h = torch.relu(update(torch.cat([h, incoming, outgoing], dim=1)))
        if not candidates:
            return h.new_empty(0)
        first = torch.tensor([r.first for r in candidates])
        second = torch.tensor([r.second for r in candidates])
        context = h.mean(dim=0).expand(len(candidates), -1)
        extra = torch.tensor([[float(r.rule == 'cancel_inverse'), float(r.rule == 'merge_rotation'),
                               float(r.rule == 'remove_identity'), rewrite_gain(circuit, r) / 10]
                              for r in candidates])
        return self.score(torch.cat([h[first], h[second], context, extra], dim=1)).squeeze(-1)

    def select(self, circuit, candidates):
        self.eval()
        with torch.no_grad():
            return candidates[int(self(circuit, candidates).argmax())]


def save_gnn(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema': SCHEMA, 'hidden': model.hidden, 'layers': model.layers,
                'state_dict': model.state_dict(), 'metadata': metadata}, path)


def load_gnn(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('Unsupported GNN checkpoint schema; retrain with this version.')
    model = RewriteGNN(checkpoint['hidden'], checkpoint['layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['metadata']
