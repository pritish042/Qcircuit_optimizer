import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .encoding import (
    encode_circuit_for_cnn,
    create_valid_gate_mask,
    MAX_GATES,
    CNN_FEATURE_SIZE
)

from .optimizer import get_redundancy_mask


# ============================================================
# 1D CNN GATE OPTIMIZER
# ============================================================

class GateOptimizerNN(nn.Module):
    """
    A 1D Convolutional Neural Network that predicts
    whether each gate should be removed or kept.

    Output for every gate position:

    1 = REMOVE
    0 = KEEP

    Example:

    Circuit:
    X X H Z Z

    Desired output:
    1 1 0 1 1

    We keep the class name GateOptimizerNN so that
    the existing main.py does not need to change.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # CNN layers
        # ----------------------------------------------------
        #
        # Input:
        #
        # (batch_size, CNN_FEATURE_SIZE, MAX_GATES)
        #
        # The CNN scans neighbouring gate positions.
        #
        # kernel_size = 3 means it examines small local
        # regions of the quantum circuit.
        #
        # padding = 1 keeps the sequence length unchanged.
        #

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=CNN_FEATURE_SIZE,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # ----------------------------------------------------
        # Final prediction layer
        # ----------------------------------------------------
        #
        # We want ONE prediction for every gate position.
        #
        # Output shape:
        #
        # (batch_size, 1, MAX_GATES)
        #

        self.output_layer = nn.Conv1d(
            in_channels=32,
            out_channels=1,
            kernel_size=1
        )


    def forward(self, x):
        """
        Forward pass through the CNN.

        Incoming shape:

        (batch, MAX_GATES, CNN_FEATURE_SIZE)

        Conv1D expects:

        (batch, features, sequence_length)

        Therefore we transpose the dimensions.
        """

        # Change:
        #
        # (batch, gates, features)
        #
        # into:
        #
        # (batch, features, gates)

        x = x.permute(
            0,
            2,
            1
        )

        # Extract local gate patterns
        x = self.cnn(x)

        # Produce one output per gate
        logits = self.output_layer(x)

        # Current shape:
        #
        # (batch, 1, MAX_GATES)
        #
        # Remove the unnecessary middle dimension
        #
        # Result:
        #
        # (batch, MAX_GATES)

        logits = logits.squeeze(1)

        return logits


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_gate_optimizer_data(circuits):
    """
    Convert quantum circuits into data that can
    be used to train the CNN.

    Returns:

    X
    =
    CNN circuit encodings

    y
    =
    correct gate-removal masks

    valid
    =
    tells us which positions are real gates
    and which positions are padding
    """

    X = []
    y = []
    valid = []


    for circuit in circuits:

        # ----------------------------------------------------
        # Encode gate type + qubit information
        # ----------------------------------------------------

        encoded = encode_circuit_for_cnn(
            circuit
        )


        # ----------------------------------------------------
        # Get correct answer from rule-based optimizer
        # ----------------------------------------------------

        mask = get_redundancy_mask(
            circuit
        )


        # ----------------------------------------------------
        # Limit mask to MAX_GATES
        # ----------------------------------------------------

        padded_mask = mask[
            :MAX_GATES
        ]


        # ----------------------------------------------------
        # Pad target mask
        # ----------------------------------------------------

        while len(padded_mask) < MAX_GATES:

            padded_mask.append(0)


        # ----------------------------------------------------
        # Create valid gate mask
        # ----------------------------------------------------

        valid_mask = create_valid_gate_mask(
            circuit
        )


        # ----------------------------------------------------
        # Store everything
        # ----------------------------------------------------

        X.append(
            encoded
        )

        y.append(
            padded_mask
        )

        valid.append(
            valid_mask
        )


    return (
        np.array(
            X,
            dtype=np.float32
        ),

        np.array(
            y,
            dtype=np.float32
        ),

        np.array(
            valid,
            dtype=np.float32
        )
    )


# ============================================================
# TRAIN CNN
# ============================================================

def train_gate_optimizer(model, X, y, valid, epochs=200, batch_size=64, seed=42):
    """Train using mini-batches, excluding padded positions from the loss."""
    if len(X) == 0 or batch_size < 1 or epochs < 1:
        raise ValueError("Training needs data, positive batch_size and positive epochs")
    X, y, valid = [torch.as_tensor(a, dtype=torch.float32) for a in (X, y, valid)]
    if X.shape[:2] != y.shape or y.shape != valid.shape:
        raise ValueError("Training input, target and padding-mask shapes disagree")
    dataset = torch.utils.data.TensorDataset(X, y, valid)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for epoch in range(epochs):
        epoch_loss, count = 0.0, 0.0
        for batch_x, batch_y, batch_valid in loader:
            raw_loss = criterion(model(batch_x), batch_y) * batch_valid
            denominator = batch_valid.sum().clamp_min(1)
            loss = raw_loss.sum() / denominator
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(raw_loss.detach().sum())
            count += float(batch_valid.sum())
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | CNN loss = {epoch_loss / max(1, count):.4f}")
    return model


# ============================================================
# PREDICT GATE MASK
# ============================================================

def predict_gate_mask(
    model,
    circuit,
    threshold=0.60
):
    """
    Predict which gates should be removed.

    Returns:

    mask

    and

    probabilities


    Example:

    Probabilities:

    [0.95, 0.91, 0.08, 0.93, 0.96]

    becomes:

    [1, 1, 0, 1, 1]
    """

    # Tell model we are predicting,
    # not training

    model.eval()


    # --------------------------------------------------------
    # Encode circuit
    # --------------------------------------------------------

    encoded = encode_circuit_for_cnn(
        circuit
    )


    # --------------------------------------------------------
    # Convert into PyTorch tensor
    # --------------------------------------------------------

    x = torch.tensor(
        encoded,
        dtype=torch.float32
    )


    # Add batch dimension
    #
    # Before:
    #
    # (MAX_GATES, CNN_FEATURE_SIZE)
    #
    # After:
    #
    # (1, MAX_GATES, CNN_FEATURE_SIZE)

    x = x.unsqueeze(0)


    # --------------------------------------------------------
    # Disable gradient calculation
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(x)


        # Convert logits into probabilities
        #
        # Example:
        #
        # 2.5  -> 0.924
        # -2.0 -> 0.119

        probabilities = torch.sigmoid(
            logits
        )


    # --------------------------------------------------------
    # Convert PyTorch tensor into NumPy
    # --------------------------------------------------------

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Only use positions containing real gates
    # --------------------------------------------------------

    gate_count = min(
        len(circuit.data),
        MAX_GATES
    )


    probabilities = probabilities[
        :gate_count
    ]


    # --------------------------------------------------------
    # Convert probabilities into binary decisions
    # --------------------------------------------------------

    mask = [
        1 if probability >= threshold else 0
        for probability in probabilities
    ]


    return (
        mask,
        probabilities
    )


# ============================================================
# ML CIRCUIT OPTIMIZER
# ============================================================

def ml_optimize_circuit(
    circuit,
    model,
    threshold=0.60
):
    """
    Use the trained CNN to create a new
    optimized quantum circuit.

    mask = 1
    means remove gate.

    mask = 0
    means keep gate.
    """

    # --------------------------------------------------------
    # Ask CNN which gates should be removed
    # --------------------------------------------------------

    mask, probabilities = predict_gate_mask(
        model,
        circuit,
        threshold
    )


    # --------------------------------------------------------
    # Create empty circuit with same structure
    # --------------------------------------------------------

    optimized = circuit.copy_empty_like()


    # --------------------------------------------------------
    # Process every original gate
    # --------------------------------------------------------

    for index, instruction in enumerate(
        circuit.data
    ):

        # Safety:
        # If circuit is larger than MAX_GATES,
        # keep everything after MAX_GATES.

        if index >= len(mask):

            keep_gate = True

        else:

            keep_gate = (
                mask[index] == 0
            )


        # ----------------------------------------------------
        # Keep gates predicted as non-redundant
        # ----------------------------------------------------

        if keep_gate:

            # Find corresponding qubits
            # in the NEW circuit.

            new_qubits = []

            for qubit in instruction.qubits:

                qubit_index = (
                    circuit
                    .find_bit(qubit)
                    .index
                )

                new_qubits.append(
                    optimized.qubits[
                        qubit_index
                    ]
                )


            # Find corresponding classical bits
            # if there are any.

            new_clbits = []

            for clbit in instruction.clbits:

                clbit_index = (
                    circuit
                    .find_bit(clbit)
                    .index
                )

                new_clbits.append(
                    optimized.clbits[
                        clbit_index
                    ]
                )


            # Add gate to optimized circuit

            optimized.append(
                instruction.operation,
                new_qubits,
                new_clbits
            )


    return (
        optimized,
        mask,
        probabilities
    )


# ============================================================
# SIMPLE GATE-LEVEL ACCURACY
# ============================================================

def calculate_gate_accuracy(
    predicted_mask,
    actual_mask
):
    """
    Calculate how many individual gate decisions
    were correct.

    Example:

    Correct:
    [1,1,0,1,1]

    Predicted:
    [1,1,0,0,1]

    4 out of 5 correct
    =
    80% accuracy
    """

    length = min(
        len(predicted_mask),
        len(actual_mask)
    )


    if length == 0:

        return 1.0


    correct = 0


    for index in range(length):

        if (
            predicted_mask[index]
            == actual_mask[index]
        ):

            correct += 1


    accuracy = (
        correct
        / length
    )


    return accuracy