"""Train a small experimental rewrite ranker. Run as python -m scripts.train_gnn."""
import argparse
import json
import random

import torch

from src.experiment_data import make_family, circuit_fingerprint
from src.graph_optimizer import find_rewrites, apply_rewrite, optimize_graph, rewrite_gain
from src.gnn_optimizer import RewriteGNN, save_gnn


def build_examples(circuits):
    examples = []
    for circuit in circuits:
        # Collect states as well as initial circuits, without using benchmark data.
        current = circuit
        for _ in range(4):
            candidates = find_rewrites(current)
            if not candidates:
                break
            rewards = []
            for candidate in candidates:
                changed = apply_rewrite(current, candidate)
                finished, _ = optimize_graph(changed, max_steps=3)
                before = sum(5 if len(i.qubits) == 2 else 1 for i in current.data)
                after = sum(5 if len(i.qubits) == 2 else 1 for i in finished.data)
                rewards.append((before - after) / 40)
            examples.append((current, candidates, torch.tensor(rewards)))
            current = apply_rewrite(current, max(candidates, key=lambda r: rewrite_gain(current, r)))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='artifacts/gnn.pt')
    args = parser.parse_args()
    if args.samples < 4 or args.epochs < 1:
        parser.error('Use at least four samples and one epoch')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    families = ('adjacent', 'interleaved', 'rotations')
    circuits = [make_family(families[i % 3], args.seed + i, blocks=4) for i in range(args.samples)]
    fingerprints = [circuit_fingerprint(qc) for qc in circuits]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('Duplicate training circuits')
    examples = build_examples(circuits)
    model = RewriteGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    order = list(range(len(examples)))
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(order)
        losses = []
        for index in order:
            circuit, candidates, targets = examples[index]
            model.train()
            loss = torch.nn.functional.mse_loss(model(circuit, candidates), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f'Epoch {epoch + 1}: training MSE={sum(losses)/len(losses):.6f}')
    metadata = {'seed': args.seed, 'epochs': args.epochs, 'samples': args.samples,
                'families': list(families), 'training_fingerprints': fingerprints,
                'objective': 'weighted gate reduction after one rewrite and three greedy steps',
                'status': 'experimental; training loss is not evidence of optimisation advantage'}
    save_gnn(model, args.output, metadata)
    print(json.dumps({'checkpoint': args.output, 'examples': len(examples)}))


if __name__ == '__main__':
    main()
