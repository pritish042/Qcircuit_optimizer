"""Compare verified outputs on seeded synthetic families, not market claims."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import torch
from qiskit import transpile

from src.circuit_utils import check_circuit_equivalence
from src.experiment_data import make_family, circuit_fingerprint
from src.graph_optimizer import optimize_graph
from src.gnn_optimizer import load_gnn
from src.optimizer import remove_redundant_gates


def circuit_metrics(circuit):
    return {'gates': sum(i.operation.name != 'barrier' for i in circuit.data),
            'two_qubit_gates': sum(len(i.qubits) == 2 and i.operation.name != 'barrier'
                                   for i in circuit.data), 'depth': circuit.depth()}


def benchmark(samples=3, seed=10000, gnn_path=None, include_cnn=False):
    torch.set_num_threads(1)
    methods = {'adjacent_rules': remove_redundant_gates,
               'graph_greedy': lambda qc: optimize_graph(qc)[0],
               # Common logical basis and no target: this is not hardware-aware routing.
               'qiskit_level_3': lambda qc: transpile(qc, optimization_level=3,
                                                     basis_gates=['u', 'cx'],
                                                     seed_transpiler=seed)}
    metadata = None
    if gnn_path:
        model, metadata = load_gnn(gnn_path)
        methods['graph_gnn'] = lambda qc: optimize_graph(qc, model.select)[0]
    if include_cnn:
        from src.gate_optimizer import GateOptimizerNN, ml_optimize_circuit
        from src.model_utils import load_model
        cnn = load_model(GateOptimizerNN(), str(Path(__file__).resolve().parents[1] /
                                               'models/gate_optimizer_cnn.pth'))
        methods['legacy_cnn'] = lambda qc: ml_optimize_circuit(qc, cnn)[0]
    rows = []
    for family in ('adjacent', 'interleaved', 'rotations', 'entangling'):
        for index in range(samples):
            qc = make_family(family, seed + index, qubits=3, blocks=6)
            fingerprint = circuit_fingerprint(qc)
            if metadata and fingerprint in metadata['training_fingerprints']:
                raise ValueError('Benchmark circuit overlaps the training corpus')
            for name, method in methods.items():
                start = perf_counter()
                candidate = method(qc.copy())
                optimize_seconds = perf_counter() - start
                start = perf_counter()
                verification = check_circuit_equivalence(qc, candidate)
                verification_seconds = perf_counter() - start
                final = candidate if verification['equivalent'] else qc
                # Common-basis metrics avoid treating a composite gate as one
                # elementary instruction, and are reported separately from raw counts.
                normalized = [transpile(c, basis_gates=['u', 'cx'], optimization_level=0,
                                        seed_transpiler=seed) for c in (qc, final)]
                rows.append({'family': family, 'seed': seed + index, 'method': name,
                             'fingerprint': fingerprint,
                             'original': circuit_metrics(qc), 'candidate': circuit_metrics(candidate),
                             'final': circuit_metrics(final),
                             'common_basis_original': circuit_metrics(normalized[0]),
                             'common_basis_final': circuit_metrics(normalized[1]),
                             'verification': verification, 'optimization_seconds': optimize_seconds,
                             'verification_seconds': verification_seconds})
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'schema': 1, 'seed': seed, 'samples_per_family': samples,
            'python': platform.python_version(), 'base_commit': commit,
            'versions': {p: importlib.metadata.version(p) for p in ('qiskit', 'torch', 'numpy')},
            'gnn_training': metadata,
            'limitations': ['Synthetic fixtures only; no quantum hardware experiments.',
                            'Entangling family is excluded from GNN training.',
                            'Legacy CNN training provenance is unavailable; no unseen-data claim for CNN.',
                            'Qiskit target is logical u/cx; no hardware performance or cost claims.',
                            'Baseline metrics do not include GNN training cost.',
                            'Working tree may contain changes beyond base_commit.'], 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--gnn')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--output', default='artifacts/benchmark.json')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    report = benchmark(args.samples, args.seed, args.gnn, args.cnn)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for method in sorted({r['method'] for r in report['rows']}):
        rows = [r for r in report['rows'] if r['method'] == method]
        before = sum(r['common_basis_original']['gates'] for r in rows)
        after = sum(r['common_basis_final']['gates'] for r in rows)
        verified = sum(r['verification']['equivalent'] for r in rows)
        print(f'{method}: {verified}/{len(rows)} verified; '
              f'common-basis accepted gate reduction={(before-after)/before:.1%}')
    print(f'Report: {destination}')


if __name__ == '__main__':
    main()
