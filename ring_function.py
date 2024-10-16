from pathlib import Path
from types import SimpleNamespace

# Federated Learning Function


def ring_function(ring_data: SimpleNamespace, secret_path: Path):
    with open(secret_path, "r") as secret_file:
        secret_value = int(secret_file.read().strip())

    return ring_data.data + secret_value
