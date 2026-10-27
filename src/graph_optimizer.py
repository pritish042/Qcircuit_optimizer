"""Dependency-aware, bounded circuit rewriting; no learned deletions.

Rewrites cross only operations on disjoint wires. We do not claim general
commutation analysis. Node indices refer to the current circuit, not a prior
iteration. Barriers are preserved and block rewrites on their wires.
"""
from dataclasses import dataclass
import math

from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, CXGate, CZGate, SwapGate,
    SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate, IGate,
)

STANDARD = {cls().name: cls for cls in (
    XGate, YGate, ZGate, HGate, CXGate, CZGate, SwapGate,
    SGate, SdgGate, TGate, TdgGate, IGate,
)}
STANDARD.update(rx=RXGate, ry=RYGate, rz=RZGate)
INVERSE = {name: name for name in ('x', 'y', 'z', 'h', 'cx', 'cz', 'swap')}
INVERSE.update(s='sdg', sdg='s', t='tdg', tdg='t')


def wires(circuit, item):
    return tuple(circuit.find_bit(q).index for q in item.qubits)


def validate_graph_input(circuit):
    """Restrict the prototype to standard gates whose semantics we know."""
    if circuit.parameters:
        raise ValueError('Bind circuit parameters before graph optimisation.')
    if len(circuit.data) > 2000:
        raise ValueError('Graph prototype is limited to 2000 instructions.')
    for item in circuit.data:
        op = item.operation
        if op.name == 'barrier':
            continue
        if item.clbits or op.name not in STANDARD or op.base_class is not STANDARD[op.name]:
            raise ValueError(f'Unsupported graph instruction: {op.name}')
        if any(not math.isfinite(float(p)) for p in op.params):
            raise ValueError('Gate parameters must be finite.')


def dependency_edges(circuit):
    """Edges (source, destination, source port, destination port), one per wire."""
    last = {}
    edges = []
    for i, item in enumerate(circuit.data):
        for port, qubit in enumerate(wires(circuit, item)):
            if qubit in last:
                source, source_port = last[qubit]
                edges.append((source, i, source_port, port))
            last[qubit] = (i, port)
    return edges


@dataclass(frozen=True)
class Rewrite:
    rule: str
    first: int
    second: int
    angle: float | None = None

    @property
    def removed(self):
        return 1 if self.rule in ('merge_rotation', 'remove_identity') else 2


def find_rewrites(circuit):
    validate_graph_input(circuit)
    candidates = []
    last = {}
    for i, item in enumerate(circuit.data):
        op = item.operation
        qs = wires(circuit, item)
        previous = {last.get(q) for q in qs}
        if op.name == 'id':
            candidates.append(Rewrite('remove_identity', i, i))
        if len(previous) == 1 and None not in previous:
            j = next(iter(previous))
            earlier = circuit.data[j]
            # Ordered ports matter: CX(a,b) is not CX(b,a).
            if wires(circuit, earlier) == qs and op.name != 'barrier':
                if INVERSE.get(op.name) == earlier.operation.name:
                    candidates.append(Rewrite('cancel_inverse', j, i))
                elif op.name in ('rx', 'ry', 'rz') and earlier.operation.name == op.name:
                    angle = float(earlier.operation.params[0]) + float(op.params[0])
                    if math.isfinite(angle):
                        candidates.append(Rewrite('merge_rotation', j, i, angle))
        for q in qs:
            last[q] = i
    return candidates


def apply_rewrite(circuit, rewrite):
    if rewrite not in find_rewrites(circuit):
        raise ValueError('Rewrite is stale or not applicable to this circuit.')
    result = circuit.copy_empty_like()
    for i, item in enumerate(circuit.data):
        if i == rewrite.first:
            if rewrite.rule == 'merge_rotation':
                op = item.operation.base_class(rewrite.angle)
                result.append(op, [result.qubits[q] for q in wires(circuit, item)])
            continue
        if i == rewrite.second:
            continue
        result.append(item.operation, [result.qubits[q] for q in wires(circuit, item)])
    return result


def rewrite_gain(circuit, rewrite):
    """Transparent gate-count objective, weighted toward two-qubit gates."""
    arity = len(circuit.data[rewrite.first].qubits)
    return rewrite.removed * (5 if arity == 2 else 1)


def optimize_graph(circuit, selector=None, max_steps=500):
    if max_steps < 0:
        raise ValueError('max_steps must be nonnegative')
    validate_graph_input(circuit)
    current = circuit.copy()
    trace = []
    for _ in range(max_steps):
        candidates = find_rewrites(current)
        if not candidates:
            break
        if selector is None:
            chosen = max(candidates, key=lambda r: rewrite_gain(current, r))
        else:
            chosen = selector(current, candidates)
        updated = apply_rewrite(current, chosen)
        trace.append({'rule': chosen.rule, 'positions': [chosen.first, chosen.second],
                      'gates_before': len(current.data), 'gates_after': len(updated.data)})
        current = updated
    return current, trace
