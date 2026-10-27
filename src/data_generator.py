# ============================================================
# SYNTHETIC QUANTUM CIRCUIT DATA GENERATOR
#
# Generates diverse quantum circuits for ML training.
#
# Goals:
# - Multiple qubits
# - Multiple gate types
# - Redundant gate pairs
# - Non-redundant gates
# - CX interactions
# - Variable circuit lengths
# ============================================================

import random
import numpy as np

from qiskit import QuantumCircuit

from .circuit_utils import circuit_to_features
from .optimizer import get_redundancy_mask


# ============================================================
# SETTINGS
# ============================================================

SINGLE_QUBIT_GATES = [
    "x",
    "h",
    "z",
    "y"
]

SELF_INVERSE_SINGLE_GATES = [
    "x",
    "h",
    "z",
    "y"
]

MIN_QUBITS = 1
MAX_QUBITS = 4

MIN_STEPS = 5
MAX_STEPS = 25


# ============================================================
# APPLY SINGLE-QUBIT GATE
# ============================================================

def apply_single_gate(
    circuit,
    gate_name,
    qubit
):
    """
    Apply a supported single-qubit gate.
    """

    if gate_name == "x":
        circuit.x(qubit)

    elif gate_name == "h":
        circuit.h(qubit)

    elif gate_name == "z":
        circuit.z(qubit)

    elif gate_name == "y":
        circuit.y(qubit)

    else:
        raise ValueError(
            f"Unsupported gate: {gate_name}"
        )


# ============================================================
# ADD REDUNDANT SINGLE-QUBIT PAIR
# ============================================================

def add_redundant_single_pair(
    circuit
):
    """
    Add a mathematically cancelling pair.

    Example:

    X(q0)
    X(q0)

    or:

    H(q1)
    H(q1)
    """

    gate_name = random.choice(
        SELF_INVERSE_SINGLE_GATES
    )

    qubit = random.randrange(
        circuit.num_qubits
    )

    apply_single_gate(
        circuit,
        gate_name,
        qubit
    )

    apply_single_gate(
        circuit,
        gate_name,
        qubit
    )


# ============================================================
# ADD REDUNDANT CX PAIR
# ============================================================

def add_redundant_cx_pair(
    circuit
):
    """
    Add two identical CX gates.

    CX is self-inverse:

    CX · CX = I
    """

    if circuit.num_qubits < 2:
        return False

    control = random.randrange(
        circuit.num_qubits
    )

    target = random.randrange(
        circuit.num_qubits
    )

    while target == control:
        target = random.randrange(
            circuit.num_qubits
        )

    circuit.cx(
        control,
        target
    )

    circuit.cx(
        control,
        target
    )

    return True


# ============================================================
# ADD NORMAL SINGLE-QUBIT GATE
# ============================================================

def add_normal_single_gate(
    circuit
):
    """
    Add a normal non-forced single-qubit gate.
    """

    gate_name = random.choice(
        SINGLE_QUBIT_GATES
    )

    qubit = random.randrange(
        circuit.num_qubits
    )

    apply_single_gate(
        circuit,
        gate_name,
        qubit
    )


# ============================================================
# ADD NORMAL CX GATE
# ============================================================

def add_normal_cx_gate(
    circuit
):
    """
    Add one CX interaction.
    """

    if circuit.num_qubits < 2:
        return False

    control = random.randrange(
        circuit.num_qubits
    )

    target = random.randrange(
        circuit.num_qubits
    )

    while target == control:
        target = random.randrange(
            circuit.num_qubits
        )

    circuit.cx(
        control,
        target
    )

    return True


# ============================================================
# ADD DISTRACTOR PATTERN
# ============================================================

def add_distractor_pattern(
    circuit
):
    """
    Add patterns that look somewhat similar
    but should NOT automatically cancel.

    These are important so the CNN does not
    simply learn:

    "same gate type = remove"

    Example:

    X(q0)
    X(q1)

    These are the same gate but different qubits,
    so they should not be treated as a cancelling pair.
    """

    if circuit.num_qubits < 2:
        add_normal_single_gate(
            circuit
        )
        return

    gate_name = random.choice(
        SELF_INVERSE_SINGLE_GATES
    )

    q1 = random.randrange(
        circuit.num_qubits
    )

    q2 = random.randrange(
        circuit.num_qubits
    )

    while q2 == q1:
        q2 = random.randrange(
            circuit.num_qubits
        )

    apply_single_gate(
        circuit,
        gate_name,
        q1
    )

    apply_single_gate(
        circuit,
        gate_name,
        q2
    )


# ============================================================
# ADD MIXED LOCAL PATTERN
# ============================================================

def add_mixed_pattern(
    circuit
):
    """
    Add a small mixed sequence.

    Example:

    H(q0)
    X(q1)
    Z(q0)

    This helps create more realistic-looking
    local contexts around redundant pairs.
    """

    steps = random.randint(
        2,
        4
    )

    for _ in range(steps):

        if (
            circuit.num_qubits >= 2
            and random.random() < 0.25
        ):
            add_normal_cx_gate(
                circuit
            )

        else:
            add_normal_single_gate(
                circuit
            )


# ============================================================
# RANDOM CIRCUIT GENERATOR
# ============================================================

def random_circuit(
    min_qubits=MIN_QUBITS,
    max_qubits=MAX_QUBITS,
    min_steps=MIN_STEPS,
    max_steps=MAX_STEPS,
    max_gates=50
):
    """
    Generate a more diverse synthetic circuit.

    The generator intentionally mixes:

    - redundant single-qubit pairs
    - redundant CX pairs
    - normal single gates
    - normal CX gates
    - distractor patterns
    - mixed gate sequences

    This creates a harder and more realistic
    ML training distribution.
    """

    if max_gates < 1:
        raise ValueError("max_gates must be positive")

    num_qubits = random.randint(
        min_qubits,
        max_qubits
    )

    circuit = QuantumCircuit(
        num_qubits
    )

    number_of_steps = random.randint(
        min_steps,
        max_steps
    )

    for _ in range(number_of_steps):
        previous_length = len(circuit.data)

        choice = random.random()

        # ----------------------------------------------------
        # 30% chance:
        # redundant single-qubit pair
        # ----------------------------------------------------

        if choice < 0.30:

            add_redundant_single_pair(
                circuit
            )

        # ----------------------------------------------------
        # 15% chance:
        # redundant CX pair
        # ----------------------------------------------------

        elif choice < 0.45:

            if not add_redundant_cx_pair(
                circuit
            ):
                add_redundant_single_pair(
                    circuit
                )

        # ----------------------------------------------------
        # 20% chance:
        # normal single-qubit gate
        # ----------------------------------------------------

        elif choice < 0.65:

            add_normal_single_gate(
                circuit
            )

        # ----------------------------------------------------
        # 15% chance:
        # normal CX
        # ----------------------------------------------------

        elif choice < 0.80:

            if not add_normal_cx_gate(
                circuit
            ):
                add_normal_single_gate(
                    circuit
                )

        # ----------------------------------------------------
        # 10% chance:
        # distractor pattern
        # ----------------------------------------------------

        elif choice < 0.90:

            add_distractor_pattern(
                circuit
            )

        # ----------------------------------------------------
        # 10% chance:
        # mixed local pattern
        # ----------------------------------------------------

        else:

            add_mixed_pattern(
                circuit
            )

        # Roll back a whole generated block rather than cutting a gate pair.
        if len(circuit.data) > max_gates:
            while len(circuit.data) > previous_length:
                circuit.data.pop()
            break

    return circuit


# ============================================================
# CREATE SIMPLE REDUNDANCY DATASET
# ============================================================

def create_dataset(
    samples=1000
):
    """
    Create training data for the simple
    redundancy-count predictor.

    X:
    numerical circuit features

    y:
    number of removable gates
    """

    X = []
    y = []

    for _ in range(samples):

        circuit = random_circuit()

        features = circuit_to_features(
            circuit
        )

        mask = get_redundancy_mask(
            circuit
        )

        removable_count = sum(
            mask
        )

        X.append(
            features
        )

        y.append(
            removable_count
        )

    return (
        np.array(
            X,
            dtype=np.float32
        ),
        np.array(
            y,
            dtype=np.float32
        )
    )