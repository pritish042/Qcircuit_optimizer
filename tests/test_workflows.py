from pathlib import Path
import random

import pytest
import torch
from qiskit import QuantumCircuit
from streamlit.testing.v1 import AppTest

from src.data_generator import random_circuit
from src.encoding import encode_circuit_for_cnn
from src.gate_optimizer import GateOptimizerNN, prepare_gate_optimizer_data, train_gate_optimizer
from scripts.benchmark import benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_cnn_input_limit_is_explicit_and_generator_respects_it():
    qc = QuantumCircuit(1)
    for _ in range(51): qc.x(0)
    with pytest.raises(ValueError, match='truncation'):
        encode_circuit_for_cnn(qc)
    random.seed(7)
    circuits = [random_circuit(min_steps=25, max_steps=25) for _ in range(20)]
    assert all(len(qc.data) <= 50 for qc in circuits)
    x, y, valid = prepare_gate_optimizer_data(circuits)
    torch.set_num_threads(1)
    trained = train_gate_optimizer(GateOptimizerNN(), x, y, valid, epochs=1, batch_size=7)
    assert all(torch.isfinite(p).all() for p in trained.parameters())


def test_benchmark_has_verified_finals_and_explicit_assumptions():
    report = benchmark(samples=1)
    assert len(report['rows']) == 12
    for row in report['rows']:
        assert row['verification']['status'] == 'equivalent'
        assert 'common_basis_final' in row
        assert row['optimization_seconds'] >= 0
    assert report['limitations']


def test_graph_page_runs_and_optimizes():
    app = AppTest.from_file(str(ROOT / 'pages/1_Graph_Lab.py'), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.success
    assert app.table


def test_legacy_cnn_page_loads():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert not app.error
