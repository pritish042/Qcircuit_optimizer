# Quantum Circuit Optimizer

A learning and research prototype comparing verified quantum-circuit rewrites,
a legacy CNN gate-deletion model, and an experimental graph neural network.

## Current capabilities

- Dependency-aware inverse cancellation across unrelated qubit operations.
- Same-axis rotation merging and identity removal.
- A greedy selector and a PyTorch message-passing GNN selecting legal rewrites.
- Bounded numerical operator verification up to global phase (maximum 8 qubits).
- Seeded comparisons with adjacent rules, the saved CNN, and Qiskit level 3.
- A Streamlit Graph Lab with circuit comparison, rewrite trace and report export.

**The first experiment does not show a GNN advantage:** GNN and greedy tied;
Qiskit achieved better reductions on the synthetic suite. See
[the audit, results and improvement plan](docs/IMPROVEMENT_PLAN.md).

## Setup

Python 3.12 is the tested runtime. Create a virtual environment and install:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements-tested.txt` records the full dependency snapshot used for the
initial development run on macOS ARM64. It is a reproducibility reference,
not a guarantee that every pinned dependency supports every platform.

## Run the application

```sh
python -m streamlit run app.py
```

Open **Graph Lab** through the sidebar or the link above the circuit inputs.
The original CNN interface remains available. Graph greedy runs without any
trained GNN checkpoint.

## Train the experimental GNN

```sh
python -m scripts.train_gnn --samples 24 --epochs 5 --seed 42
```

This writes `artifacts/gnn.pt`. Restart or rerun Graph Lab to select the GNN.
Training uses adjacent, interleaved and rotation families. Targets describe
weighted gate reduction after one candidate rewrite and three greedy steps;
they are not globally optimal labels. Training loss is not a quality benchmark.

## Compare optimisers

```sh
python -m scripts.benchmark --samples 4 --seed 10000 --gnn artifacts/gnn.pt --cnn
```

The report is saved to `artifacts/benchmark.json`. Omit `--gnn` or `--cnn` to
run without those models. Each candidate is verified. Unverified candidates
are rejected and the original circuit is used for accepted-output metrics.
Reports include raw counts and counts normalised to a logical `u,cx` basis.
There is no hardware target or quantum-hardware execution in this benchmark.

The entangling family is held out from GNN training. Exact training fingerprint
overlap is rejected. All families are synthetic and do not establish performance
on real applications. The saved CNN's training corpus is not recorded; do not
claim its benchmark inputs are unseen.

## Tests

```sh
python -m pytest -q
python test_equivalence.py
```

Tests cover dependencies, directed CX operands, barriers, rotations, global
phase, unsupported input, teacher consistency, training, checkpoints, benchmark
reports and Streamlit interaction.

## Supported scope

Graph rewrites support standard X/Y/Z/H, S/Sdg, T/Tdg, CX/CZ/SWAP, RX/RY/RZ,
identity and barriers. Angles must be bound and finite. Barriers are retained
and block rewrites on their wires. Only disjoint-wire commutation is supported;
there is no general commutation analysis. Graph inputs are limited to 2,000
instructions and the default rewrite budget is 500 steps.

Numerical verification is bounded to eight qubits because full unitary matrices
scale exponentially. Its results are `equivalent`, `not_equivalent`, or
`inconclusive`. Only `equivalent` candidates can be accepted. Numerical agreement
within tolerance is not a symbolic proof for arbitrary sizes or parameters.
Measurements, resets and dynamic control flow are outside the current scope.

The saved CNN supports at most four qubits and 50 gate positions. Oversized
inputs now raise an error instead of being silently truncated. Its original
metrics file is historical and predates the corrected teacher. No fallback
performance numbers are invented when metrics are missing.

## Layout

- `src/graph_optimizer.py`: dependency edges and legal rewrite engine.
- `src/gnn_optimizer.py`: message-passing rewrite ranker and checkpoints.
- `src/experiment_data.py`: seeded synthetic circuit families.
- `scripts/train_gnn.py`: reproducible prototype training.
- `scripts/benchmark.py`: cross-method verified evaluation.
- `pages/1_Graph_Lab.py`: interactive experiment interface.
- `src/circuit_utils.py`: bounded operator verification and legacy helpers.
- `src/gate_optimizer.py`: legacy CNN with mini-batch training.
- `main.py`: legacy multi-model training/demo workflow.

OKF and NCM are not implemented; see the improvement plan for why their role
must be specified and evaluated before integration.

## Team

Pritish Dash — Team Leader

Tarun Ku. Choudhury — Team Member
