"""
Schedule action formatting and JSON writer.
Centralizes action schema in one place.
"""

import json
from typing import Dict, List


def emit_load(tensor_name: str) -> Dict:
    """Emit a load action."""
    return {'type': 'load', 'tensor': tensor_name}


def emit_store(tensor_name: str) -> Dict:
    """Emit a store action."""
    return {'type': 'store', 'tensor': tensor_name}


def emit_compute(op_name: str) -> Dict:
    """Emit a compute action."""
    return {'type': 'compute', 'op': op_name}


def emit_free(tensor_name: str) -> Dict:
    """Emit a free action (experimental)."""
    return {'type': 'free', 'tensor': tensor_name}


def write_schedule(actions: List[Dict], filepath: str) -> None:
    """Write schedule JSON to disk."""
    output = {'actions': actions}
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
