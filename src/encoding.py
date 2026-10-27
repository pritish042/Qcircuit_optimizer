import numpy as np


# ============================================================
# BASIC SETTINGS
# ============================================================

# Supported quantum gates
GATE_MAP = {
    "x": 1,
    "h": 2,
    "z": 3,
    "cx": 4,
    "y": 5,
    "s": 6,
    "t": 7,
    "rx": 8,
    "ry": 9,
    "rz": 10
}


# Maximum number of gates that we will represent
MAX_GATES = 50


# Maximum number of qubits supported by the CNN encoder
#
# Our current generated circuits use only 2 qubits,
# but 4 gives us room to expand later.
MAX_QUBITS = 4


# ============================================================
# LEGACY / AUTOENCODER ENCODING
# ============================================================

def encode_circuit(circuit):
    """
    Convert a quantum circuit into a fixed-length
    numerical vector.

    This is the ORIGINAL encoder.

    It is kept because our existing autoencoder
    currently depends on it.

    Example:

    X -> 1
    H -> 2
    Z -> 3

    Circuit:

    X X H Z Z

    becomes:

    [1, 1, 2, 3, 3, 0, 0, 0, ...]

    Length is always MAX_GATES.
    """

    vector = []

    for instruction in circuit.data:

        gate_name = instruction.operation.name

        gate_value = GATE_MAP.get(
            gate_name,
            0
        )

        vector.append(
            gate_value
        )

    # Add padding until the vector has MAX_GATES values
    while len(vector) < MAX_GATES:

        vector.append(0)

    # Cut circuits longer than MAX_GATES
    vector = vector[:MAX_GATES]

    return np.array(
        vector,
        dtype=np.float32
    )


# ============================================================
# LEGACY DECODER
# ============================================================

def decode_vector(vector):
    """
    Convert the old numerical encoding back
    into gate names.

    Example:

    [1, 1, 2, 3]

    becomes:

    ["x", "x", "h", "z"]
    """

    reverse_map = {
        value: key
        for key, value in GATE_MAP.items()
    }

    return [
        reverse_map.get(
            int(value),
            "PAD"
        )
        for value in vector
    ]


# ============================================================
# CNN FEATURE SETTINGS
# ============================================================

# Number of known gates
NUM_KNOWN_GATES = len(GATE_MAP)


# One extra feature is reserved for unsupported/unknown gates
UNKNOWN_GATE_INDEX = NUM_KNOWN_GATES


# Total number of gate-type features
NUM_GATE_FEATURES = NUM_KNOWN_GATES + 1


# Each gate position will contain:
#
# gate type features
# +
# first qubit features
# +
# second qubit features

CNN_FEATURE_SIZE = (
    NUM_GATE_FEATURES
    + MAX_QUBITS
    + MAX_QUBITS
)


# ============================================================
# CNN ENCODER
# ============================================================

def encode_circuit_for_cnn(circuit):
    """
    Convert a quantum circuit into a representation
    suitable for a 1D CNN.

    Unlike the old encoder, this stores BOTH:

    1. Gate type
    2. Which qubit(s) the gate acts on

    Output shape:

    (MAX_GATES, CNN_FEATURE_SIZE)

    Example idea:

    X(q0)
    X(q0)
    H(q1)
    Z(q0)
    Z(q0)

    Now the ML model can distinguish:

    X(q0)

    from:

    X(q1)

    which the old encoder could NOT do.
    """

    if len(circuit.data) > MAX_GATES:
        raise ValueError(f"CNN supports at most {MAX_GATES} gates; refusing silent truncation.")
    if circuit.num_qubits > MAX_QUBITS:
        raise ValueError(f"CNN supports at most {MAX_QUBITS} qubits.")

    # Start with an empty matrix filled with zeros.
    #
    # Rows = gate positions
    # Columns = gate/qubit features

    encoded = np.zeros(
        (
            MAX_GATES,
            CNN_FEATURE_SIZE
        ),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Process every gate
    # --------------------------------------------------------

    for position, instruction in enumerate(
        circuit.data
    ):

        # Stop if circuit is longer than MAX_GATES
        if position >= MAX_GATES:
            break


        # ====================================================
        # 1. ENCODE GATE TYPE
        # ====================================================

        gate_name = instruction.operation.name


        if gate_name in GATE_MAP:

            # GATE_MAP starts from 1.
            #
            # NumPy positions start from 0.
            #
            # Therefore:
            #
            # X = 1 -> index 0
            # H = 2 -> index 1
            # Z = 3 -> index 2

            gate_index = (
                GATE_MAP[gate_name]
                - 1
            )

        else:

            # Unknown gate gets its own feature
            gate_index = UNKNOWN_GATE_INDEX


        encoded[
            position,
            gate_index
        ] = 1.0


        # ====================================================
        # 2. FIND QUBIT NUMBERS
        # ====================================================

        qubit_indices = []

        for qubit in instruction.qubits:

            qubit_index = (
                circuit
                .find_bit(qubit)
                .index
            )

            qubit_indices.append(
                qubit_index
            )


        # ====================================================
        # 3. VALIDATE QUBIT RANGE
        # ====================================================

        for qubit_index in qubit_indices:

            if qubit_index >= MAX_QUBITS:

                raise ValueError(
                    f"Circuit uses qubit {qubit_index}, "
                    f"but MAX_QUBITS is {MAX_QUBITS}. "
                    f"Increase MAX_QUBITS in encoding.py."
                )


        # ====================================================
        # 4. ENCODE FIRST QUBIT
        # ====================================================

        if len(qubit_indices) >= 1:

            first_qubit = qubit_indices[0]

            first_qubit_offset = (
                NUM_GATE_FEATURES
            )

            encoded[
                position,
                first_qubit_offset
                + first_qubit
            ] = 1.0


        # ====================================================
        # 5. ENCODE SECOND QUBIT
        # ====================================================

        if len(qubit_indices) >= 2:

            second_qubit = qubit_indices[1]

            second_qubit_offset = (
                NUM_GATE_FEATURES
                + MAX_QUBITS
            )

            encoded[
                position,
                second_qubit_offset
                + second_qubit
            ] = 1.0


    return encoded


# ============================================================
# CNN ENCODING DEBUGGER
# ============================================================

def describe_cnn_encoding(
    circuit,
    encoded=None
):
    """
    Print a simple human-readable explanation
    of the CNN encoding.

    Useful while debugging the ML model.
    """

    if encoded is None:

        encoded = encode_circuit_for_cnn(
            circuit
        )


    print("\nCNN CIRCUIT ENCODING")
    print("====================")


    for position, instruction in enumerate(
        circuit.data
    ):

        if position >= MAX_GATES:
            break


        gate_name = (
            instruction
            .operation
            .name
            .upper()
        )


        qubits = []

        for qubit in instruction.qubits:

            qubit_index = (
                circuit
                .find_bit(qubit)
                .index
            )

            qubits.append(
                qubit_index
            )


        print(
            f"Position {position}: "
            f"{gate_name} "
            f"on qubit(s) {qubits}"
        )


# ============================================================
# VALID POSITION MASK
# ============================================================

def create_valid_gate_mask(circuit):
    """
    Create a mask identifying real gate positions.

    Example:

    Circuit has 5 gates.

    Result:

    [1,1,1,1,1,0,0,0,...]

    1 = actual gate
    0 = padding
    """

    mask = np.zeros(
        MAX_GATES,
        dtype=np.float32
    )


    gate_count = min(
        len(circuit.data),
        MAX_GATES
    )


    mask[:gate_count] = 1.0


    return mask