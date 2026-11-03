
"""Dataset of independent, fixed-length SFT conversation windows."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np


class SFTSequenceDataset:
    def __init__(
        self,
        token_path: str | Path,
        mask_path: str | Path,
        context_len: int,
        seed: int,
    ):
        if context_len < 1:
            raise ValueError("context_len must be positive.")

        self.token_path = Path(token_path)
        self.mask_path = Path(mask_path)
        self.context_len = int(context_len)
        self.sequence_len = self.context_len + 1
        self.seed = int(seed)

        token_bytes = self.token_path.stat().st_size
        mask_bytes = self.mask_path.stat().st_size

        token_row_bytes = self.sequence_len * np.dtype(np.uint16).itemsize
        mask_row_bytes = self.sequence_len * np.dtype(np.uint8).itemsize

        if token_bytes == 0 or mask_bytes == 0:
            raise ValueError("SFT token and mask files must not be empty.")

        if token_bytes % token_row_bytes != 0:
            raise ValueError(
                "Token file size is not aligned to context_len + 1."
            )

        if mask_bytes % mask_row_bytes != 0:
            raise ValueError(
                "Mask file size is not aligned to context_len + 1."
            )

        token_rows = token_bytes // token_row_bytes
        mask_rows = mask_bytes // mask_row_bytes

        if token_rows != mask_rows:
            raise ValueError(
                f"Token/mask row mismatch: {token_rows} != {mask_rows}."
            )

        self.num_sequences = int(token_rows)

        if self.num_sequences < 1:
            raise ValueError("Dataset contains no sequences.")

        self.tokens = np.memmap(
            self.token_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_sequences, self.sequence_len),
        )
        self.masks = np.memmap(
            self.mask_path,
            dtype=np.uint8,
            mode="r",
            shape=(self.num_sequences, self.sequence_len),
        )

        self.epoch = 0
        self.cursor = 0
        self.order = np.empty(0, dtype=np.int64)
        self._shuffle()

    def __len__(self) -> int:
        return self.num_sequences

    def _shuffle(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.order = rng.permutation(self.num_sequences).astype(np.int64)

    def _sequence(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(self.tokens[int(index)], dtype=np.int32)
        # Loss is applied to targets tokens[:, 1:], not inputs tokens[:, :-1].
        mask = np.asarray(self.masks[int(index), 1:], dtype=np.float32)
        return tokens, mask

    def next_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        indices: list[int] = []

        while len(indices) < batch_size:
            remaining = self.num_sequences - self.cursor
            needed = batch_size - len(indices)
            take = min(remaining, needed)

            if take:
                indices.extend(
                    self.order[self.cursor : self.cursor + take].tolist()
                )
                self.cursor += take

            if self.cursor >= self.num_sequences:
                self.epoch += 1
                self.cursor = 0
                self._shuffle()

        rows = [self._sequence(index) for index in indices]

        tokens = np.stack([row[0] for row in rows]).astype(np.int32)
        masks = np.stack([row[1] for row in rows]).astype(np.float32)

        return mx.array(tokens), mx.array(masks)

    def state(self) -> dict:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "num_sequences": self.num_sequences,
            "context_len": self.context_len,
        }

    def load_state(self, state: dict) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("Dataset seed differs from checkpoint state.")

        if int(state["num_sequences"]) != self.num_sequences:
            raise ValueError(
                "Dataset sequence count differs from checkpoint state."
            )

        if int(state["context_len"]) != self.context_len:
            raise ValueError(
                "Dataset context length differs from checkpoint state."
            )

        epoch = int(state["epoch"])
        cursor = int(state["cursor"])

        if epoch < 0 or not 0 <= cursor < self.num_sequences:
            raise ValueError("Invalid dataset epoch/cursor in checkpoint.")

        self.epoch = epoch
        self.cursor = cursor
        self._shuffle()

