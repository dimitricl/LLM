import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_checkpoint(
    path: str | Path, model, optimizer, meta: dict
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(path / "model.safetensors"),
        tree_flatten(model.parameters(), destination={}),
    )
    optimizer_state = tree_flatten(optimizer.state, destination={})
    optimizer_state = {
        key: (value.astype(mx.int32) if "step" in str(key) else value)
        for key, value in optimizer_state.items()
    }
    mx.save_safetensors(str(path / "optimizer.safetensors"), optimizer_state)
    (path / "state.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_checkpoint(
    path: str | Path, model, optimizer=None
) -> dict:
    path = Path(path)
    weights_path = path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    model.load_weights(list(mx.load(str(weights_path)).items()))
    if optimizer is not None:
        optimizer_state = mx.load(str(path / "optimizer.safetensors"))
        optimizer.state = optimizer_state
    return json.loads((path / "state.json").read_text())
