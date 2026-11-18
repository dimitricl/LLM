from pathlib import Path

import mlx.core as mx
import numpy as np

from tinyllm.data import read_bin


class TokenDataset:
    def __init__(
        self, path: str | Path, context_len: int, seed: int
    ) -> None:
        self.data = read_bin(path)
        self.context_len = context_len
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._num_sequences = max(0, (len(self.data) - 1) // context_len)
        self._order = np.empty(0, dtype=np.int64)
        self._shuffle()

    def _shuffle(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        self._order = rng.permutation(self._num_sequences).astype(np.int64)

    def _sequence(self, index: int) -> np.ndarray:
        start = int(index) * self.context_len
        return self.data[start : start + self.context_len + 1]

    def next_batch(self, batch_size: int) -> mx.array:
        if self.cursor + batch_size > len(self._order):
            self.epoch += 1
            self.cursor = 0
            self._shuffle()
        indices = self._order[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        rows = np.stack([self._sequence(i) for i in indices])
        return mx.array(rows.astype(np.int32))

    def state(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state(self, state: dict) -> None:
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self._shuffle()
