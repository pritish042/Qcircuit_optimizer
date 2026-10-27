# ============================================================
# QUANTUM CIRCUIT UTILITIES & EQUIVALENCE SAFETY LAYER
# ============================================================

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import numpy as np


def create_messy_circuit():
    """Create a simple messy quantum circuit with redundant gates"""
    qc = QuantumCircuit(1, name="messy")
    
    qc.x(0)   # Flip
    qc.x(0)   # Flip again (these two cancel)
    qc.h(0)   # Hadamard
    qc.x(0)   # Flip
    qc.x(0)   # Flip again (these two also cancel)
    
    return qc


def create_clean_circuit():
    """The optimized version of the messy circuit"""
    qc = QuantumCircuit(1, name="clean")
    qc.h(0)
    return qc


def get_circuit_info(qc):
    """Return basic information about a circuit"""
    depth = qc.depth()
    gates = len(qc.data)
    return depth, gates


def get_statevector(qc):
    """Simulate the circuit and return the final statevector from |0...0>."""
    return Statevector.from_instruction(qc)


def compare_circuits(qc1, qc2):
    """
    Check statevector fidelity between two circuits starting from |0...0>.

    NOTE: Statevector fidelity measured from |0...0> is useful for basic 
    state diagnostics, but is NOT mathematically sufficient to prove overall 
    circuit/operator equivalence.
    """
    try:
        state1 = get_statevector(qc1)
        state2 = get_statevector(qc2)
        fidelity = np.abs(np.vdot(state1, state2))**2
        return float(fidelity)
    except Exception:
        return 0.0


def circuit_to_features(circuit):
    """Extract summary numerical features from a quantum circuit."""
    counts = circuit.count_ops()

    num_x = counts.get("x", 0)
    num_h = counts.get("h", 0)
    num_z = counts.get("z", 0)
    num_cx = counts.get("cx", 0)

    depth = circuit.depth()
    gate_count = sum(counts.values())

    return [
        num_x,
        num_h,
        num_z,
        num_cx,
        depth,
        gate_count
    ]


# ============================================================
# EXACT OPERATOR EQUIVALENCE CHECK (FULL QUANTUM SAFETY LAYER)
# ============================================================

MAX_EXACT_QUBITS = 8


def check_circuit_equivalence(circuit1, circuit2, tolerance=1e-7,
                              max_qubits=MAX_EXACT_QUBITS) -> dict:
    """Bounded numerical operator comparison up to global phase.

    Unsupported or oversized inputs are inconclusive, never a proof of
    non-equivalence. ``equivalent`` remains false for compatibility with the
    legacy accept/reject API, so inconclusive proposals are not accepted.
    """
    from qiskit.circuit import Gate

    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-3:
        raise ValueError("tolerance must be finite and between 0 and 1e-3")
    if not isinstance(max_qubits, int) or not 1 <= max_qubits <= MAX_EXACT_QUBITS:
        raise ValueError(f"max_qubits must be between 1 and {MAX_EXACT_QUBITS}")

    def result(status, reason, error=None, phase=None):
        return {"status": status, "equivalent": status == "equivalent",
                "reason": reason, "equivalence_error": error,
                "global_phase_angle": phase, "method": "numerical_operator"}

    if circuit1.num_qubits != circuit2.num_qubits:
        return result("not_equivalent", "Qubit counts differ.")
    if circuit1.num_qubits > max_qubits:
        return result("inconclusive", f"Exact verification is limited to {max_qubits} qubits.")
    for circuit in (circuit1, circuit2):
        if circuit.parameters:
            return result("inconclusive", "Bind all parameters before verification.")
        for item in circuit.data:
            if item.operation.name == "barrier":
                continue
            if not isinstance(item.operation, Gate) or item.clbits:
                return result("inconclusive", "Only bound unitary gates and barriers are supported.")
    try:
        u1, u2 = Operator(circuit1).data, Operator(circuit2).data
    except Exception as error:
        return result("inconclusive", f"Cannot construct operators: {error}")
    if not np.isfinite(u1).all() or not np.isfinite(u2).all():
        return result("inconclusive", "Operators contain non-finite entries.")
    # Largest-magnitude entry avoids extracting phase from a tiny coefficient.
    index = np.unravel_index(np.argmax(np.abs(u1)), u1.shape)
    if abs(u1[index]) == 0:
        return result("inconclusive", "Invalid zero operator.")
    if abs(u2[index]) == 0:
        return result("not_equivalent", "Operator support differs.", float(abs(u1[index])))
    phase = u2[index] / u1[index]
    phase /= abs(phase)
    error = float(np.max(np.abs(u2 - phase * u1)))
    status = "equivalent" if error <= tolerance else "not_equivalent"
    return result(status, "Operators agree up to global phase within tolerance."
                  if status == "equivalent" else "Operators differ beyond tolerance.",
                  error, float(np.angle(phase)))
