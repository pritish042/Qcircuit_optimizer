"""Seeded circuit families for experiments; not an industrial benchmark suite."""
import hashlib
import math
import random

from qiskit import QuantumCircuit, qasm2


def circuit_fingerprint(circuit):
    return hashlib.sha256(qasm2.dumps(circuit).encode()).hexdigest()


def make_family(family, seed, qubits=3, blocks=8):
    if family not in ('adjacent', 'interleaved', 'rotations', 'entangling'):
        raise ValueError(f'Unknown family: {family}')
    if qubits < 2 or blocks < 1:
        raise ValueError('Use at least two qubits and one block')
    rng = random.Random(seed)
    qc = QuantumCircuit(qubits)
    for _ in range(blocks):
        q = rng.randrange(qubits)
        other = (q + 1) % qubits
        gate = rng.choice(('x', 'y', 'z', 'h'))
        if family == 'rotations':
            axis = rng.choice(('rx', 'ry', 'rz'))
            angle = rng.uniform(-math.pi, math.pi)
            getattr(qc, axis)(angle, q)
            qc.h(other)
            getattr(qc, axis)(rng.uniform(-math.pi, math.pi), q)
        elif family == 'entangling':
            qc.cx(q, other)
            if qubits > 2:
                qc.h((q + 2) % qubits)
            qc.cx(q, other)
        else:
            getattr(qc, gate)(q)
            if family == 'interleaved':
                qc.h(other)
            getattr(qc, gate)(q)
        # Prevent the entire fixture from being a chain of cancelling pairs.
        qc.rz(rng.uniform(-1, 1), other)
        qc.cx(other, q)
    return qc
