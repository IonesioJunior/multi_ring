from pathlib import Path
from types import SimpleNamespace
import json
import diffprivlib.tools as dp

# Federated Learning Function


def ring_function(ring_data: SimpleNamespace, secret_path: Path):
    with open(secret_path, "r") as secret_file:
        secret_data = json.load(secret_file)

    return ring_data.data + dp.mean(
        secret_data["data"],
        epsilon=float(secret_data["epsilon"]),
        bounds=(float(secret_data["bound_min"]), float(secret_data["bound_max"])),
    )
