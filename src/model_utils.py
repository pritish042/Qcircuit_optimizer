import os
import torch
import json

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path="models/gate_optimizer_cnn.pth"
):
    """
    Save the trained model weights.

    Example:
    models/gate_optimizer_cnn.pth
    """

    # Create the models folder if it doesn't exist
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    torch.save(
        model.state_dict(),
        path
    )

    print(
        f"Model saved to: {path}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model,
    path="models/gate_optimizer_cnn.pth"
):
    """
    Load previously trained model weights.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"Model loaded from: {path}"
    )

    return model


# ============================================================
# CHECK IF MODEL EXISTS
# ============================================================

def model_exists(
    path="models/gate_optimizer_cnn.pth"
):
    """
    Check whether a saved model already exists.
    """

    return os.path.exists(
        path
    )

# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

import json


def save_metrics(
    metrics,
    path="models/evaluation_metrics.json"
):
    """
    Save evaluation results to a JSON file.

    This lets the Streamlit frontend display
    the real metrics associated with the model.
    """

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True
        )

    # Convert NumPy / non-standard numeric values
    # into normal Python numbers where needed.
    clean_metrics = {}

    for key, value in metrics.items():

        try:
            clean_metrics[key] = float(value)

        except (TypeError, ValueError):
            clean_metrics[key] = value

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            clean_metrics,
            file,
            indent=4
        )

    print(
        f"Evaluation metrics saved to: {path}"
    )


# ============================================================
# LOAD EVALUATION METRICS
# ============================================================

def load_metrics(
    path="models/evaluation_metrics.json"
):
    """
    Load saved evaluation metrics.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with open(
        path,
        "r"
    ) as file:

        metrics = json.load(
            file
        )

    return metrics