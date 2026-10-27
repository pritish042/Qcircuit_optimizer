import math
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Gate

from src.circuit_utils import check_circuit_equivalence
from src.graph_optimizer import optimize_graph, dependency_edges, find_rewrites, apply_rewrite
from src.optimizer import get_redundancy_mask, remove_redundant_gates
from src.gnn_optimizer import RewriteGNN, save_gnn, load_gnn
from src.experiment_data import make_family, circuit_fingerprint


def test_nested_teacher_matches_rule_optimizer():
    qc = QuantumCircuit(1)
    qc.x(0); qc.h(0); qc.h(0); qc.x(0)
    assert get_redundancy_mask(qc) == [1, 1, 1, 1]
    assert len(remove_redundant_gates(qc).data) == 0


def test_disjoint_interleaving_and_dependency_roles():
    qc = QuantumCircuit(3)
    qc.cx(0, 1); qc.h(2); qc.cx(0, 1)
    assert dependency_edges(qc) == [(0, 2, 0, 0), (0, 2, 1, 1)]
    result, trace = optimize_graph(qc)
    assert [i.operation.name for i in result.data] == ['h']
    assert trace[0]['positions'] == [0, 2]
    assert check_circuit_equivalence(qc, result)['equivalent']


@pytest.mark.parametrize('case', ['opposite_cx', 'shared_wire', 'barrier'])
def test_dependency_conflicts_block_cancellation(case):
    qc = QuantumCircuit(2)
    if case == 'opposite_cx':
        qc.cx(0, 1); qc.cx(1, 0)
    else:
        qc.x(0)
        qc.h(0) if case == 'shared_wire' else qc.barrier(0)
        qc.x(0)
    result, trace = optimize_graph(qc)
    assert not trace
    assert len(result.data) == len(qc.data)


@pytest.mark.parametrize('axis', ['rx', 'ry', 'rz'])
def test_rotations_preserve_global_phase_and_angles(axis):
    qc = QuantumCircuit(2, global_phase=0.23)
    getattr(qc, axis)(math.pi, 0)
    qc.h(1)
    getattr(qc, axis)(2 * math.pi, 0)
    result, trace = optimize_graph(qc)
    assert len(result.data) == 2
    assert result.data[0].operation.params[0] == pytest.approx(3 * math.pi)
    assert result.global_phase == pytest.approx(qc.global_phase)
    assert check_circuit_equivalence(qc, result)['equivalent']


@pytest.mark.parametrize('gate,inverse', [('s', 'sdg'), ('t', 'tdg'), ('h', 'h'), ('y', 'y')])
def test_inverse_rules(gate, inverse):
    qc = QuantumCircuit(2)
    getattr(qc, gate)(0); qc.z(1); getattr(qc, inverse)(0)
    result, _ = optimize_graph(qc)
    assert len(result.data) == 1
    assert check_circuit_equivalence(qc, result)['equivalent']


@pytest.mark.parametrize('kind', ['measure', 'reset', 'symbolic', 'impostor'])
def test_graph_rejects_unsupported_semantics(kind):
    qc = QuantumCircuit(1, 1)
    if kind == 'measure': qc.measure(0, 0)
    if kind == 'reset': qc.reset(0)
    if kind == 'symbolic': qc.rx(Parameter('theta'), 0)
    if kind == 'impostor': qc.append(Gate('x', 1, []), [0])
    with pytest.raises(ValueError): optimize_graph(qc)


def test_stale_rewrite_is_rejected():
    qc = QuantumCircuit(1); qc.x(0); qc.x(0)
    rewrite = find_rewrites(qc)[0]
    updated = apply_rewrite(qc, rewrite)
    with pytest.raises(ValueError): apply_rewrite(updated, rewrite)


def test_verifier_distinguishes_failure_from_unsupported():
    qc = QuantumCircuit(1); qc.z(0)
    assert check_circuit_equivalence(qc, QuantumCircuit(1))['status'] == 'not_equivalent'
    qc.reset(0)
    assert check_circuit_equivalence(qc, qc)['status'] == 'inconclusive'
    assert check_circuit_equivalence(QuantumCircuit(9), QuantumCircuit(9))['status'] == 'inconclusive'
    with pytest.raises(ValueError): check_circuit_equivalence(qc, qc, tolerance=float('nan'))


def test_barriers_are_harmless_to_verification():
    qc = QuantumCircuit(1); qc.h(0); qc.barrier()
    candidate = QuantumCircuit(1); candidate.h(0); candidate.global_phase = 0.5
    assert check_circuit_equivalence(qc, candidate)['equivalent']


def test_gnn_backward_checkpoint_and_safe_selection(tmp_path):
    torch.manual_seed(3)
    qc = make_family('interleaved', 9, blocks=2)
    candidates = find_rewrites(qc)
    model = RewriteGNN()
    scores = model(qc, candidates)
    assert scores.shape == (len(candidates),)
    scores.sum().backward()
    assert model.input.weight.grad is not None
    checkpoint = tmp_path / 'model.pt'
    save_gnn(model, checkpoint, {'seed': 3})
    loaded, metadata = load_gnn(checkpoint)
    assert metadata['seed'] == 3
    assert torch.allclose(scores, loaded(qc, candidates))
    result, _ = optimize_graph(qc, loaded.select)
    assert check_circuit_equivalence(qc, result)['equivalent']


@pytest.mark.parametrize('family', ['adjacent', 'interleaved', 'rotations', 'entangling'])
def test_seeded_family_rewrites_preserve_operation(family):
    for seed in range(10):
        qc = make_family(family, seed, qubits=4, blocks=4)
        result, _ = optimize_graph(qc)
        assert check_circuit_equivalence(qc, result)['equivalent']
        assert len(result.data) <= len(qc.data)
        assert circuit_fingerprint(qc) == circuit_fingerprint(make_family(family, seed, qubits=4, blocks=4))


def test_empty_and_budgeted_circuits():
    qc = QuantumCircuit(2)
    result, trace = optimize_graph(qc)
    assert not result.data and not trace
    qc.id(0)
    assert len(optimize_graph(qc, max_steps=0)[0].data) == 1
    assert not optimize_graph(qc)[0].data


def test_original_safety_regressions():
    from test_equivalence import run_equivalence_test_suite
    run_equivalence_test_suite()
