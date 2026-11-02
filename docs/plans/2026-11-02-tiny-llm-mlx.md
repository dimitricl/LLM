# Tiny LLM from scratch en MLX — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire de zéro en MLX un LLM decoder-only de ~42 M puis ~80 M de paramètres, avec tokenizer, données, entraînement, évaluation, quantification 4-bit et CLI.

**Architecture:** Pipeline en deux phases. La phase 1 valide le pipeline sur TinyStories avec un modèle de ~42 M. La phase 2 entraîne from scratch un modèle de ~80 M sur un corpus mixte EN/FR/code. Le modèle est un transformer pre-norm style Llama (RMSNorm, RoPE, SwiGLU, GQA, embeddings liés), entraîné avec un AdamW maison en précision mixte bf16/fp32.

**Tech Stack:** Python 3.11, `uv`, MLX, NumPy, `tokenizers`, `datasets`, pytest, ruff.

**Spec:** `docs/specs/2026-11-02-tiny-llm-mlx-design.md`

---

## Carte des fichiers

| Fichier | Responsabilité |
|---|---|
| `pyproject.toml` | dépendances, pytest, ruff |
| `src/tinyllm/config.py` | dataclasses de config + chargement JSON validé |
| `src/tinyllm/tokenizer.py` | entraînement BPE, wrapper encode/decode, table d'octets |
| `src/tinyllm/data.py` | nettoyage, dédup, splits, packing, binaire uint16 |
| `src/tinyllm/prepare.py` | CLI de préparation des données |
| `src/tinyllm/model.py` | RMSNorm, RoPE, GQA, SwiGLU, TinyLM, cache KV |
| `src/tinyllm/train.py` | boucle, précision mixte, AdamW, checkpoints, logs |
| `src/tinyllm/checkpoint.py` | sauvegarde/reprise des poids et de l'état optimiseur |
| `src/tinyllm/dataset.py` | lecture memmap, batching, shuffling |
| `src/tinyllm/sample.py` | génération + CLI |
| `src/tinyllm/eval.py` | BPB, perplexité par domaine |
| `src/tinyllm/quantize.py` | quantification 4-bit, save/load, mesures |
| `tests/` | un fichier par unité ci-dessus |
| `configs/` | phase 1, phase 2, phase 2 étendue |
| `scripts/` | orchestration shell |

## Ordre des tâches

- Tâche 0 — Bootstrap du dépôt
- Tâche 1 — Configurations typées
- Tâche 2 — Tokenizer BPE byte-level
- Tâche 3 — Nettoyage, déduplication, splits
- Tâche 4 — Packing binaire uint16
- Tâche 5 — CLI de préparation des données
- Tâche 6 — RMSNorm et RoPE
- Tâche 7 — Attention causale GQA
- Tâche 8 — MLP, bloc et TinyLM
- Tâche 9 — Cache KV
- Tâche 10 — Loss, précision mixte, accumulation
- Tâche 11 — Checkpoints et reprise
- Tâche 12 — Dataset et batching
- Tâche 13 — Boucle d'entraînement
- Tâche 14 — Génération
- Tâche 15 — Évaluation BPB
- Tâche 16 — Quantification et chargement
- Tâche 17 — Configs, scripts et documentation des sources
- Tâche 18 — Exécution phase 1
- Tâche 19 — Exécution phase 2
- Tâche 20 — Livraison, mesures et rapport

---
### Tâche 0 : Bootstrap du dépôt et de l'outillage

**Fichiers :**
- Créer : `pyproject.toml`
- Créer : `.gitignore`
- Créer : `src/tinyllm/__init__.py`
- Créer : `tests/test_import.py`

- [ ] **Étape 1 : Écrire `pyproject.toml`**

```toml
[project]
name = "tinyllm"
version = "0.1.0"
description = "Tiny LLM from scratch in MLX"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.22",
    "numpy>=1.26",
    "tokenizers>=0.20",
    "datasets>=3.0",
    "huggingface-hub>=0.26",
]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.7"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tinyllm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: tests longs (overfit, smoke run)"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Étape 2 : Écrire `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
runs/
models/
data/
.pytest_cache/
.ruff_cache/
*.safetensors
*.bin
```

- [ ] **Étape 3 : Créer le paquet et un test d'import**

`src/tinyllm/__init__.py` :

```python
"""Tiny LLM from scratch in MLX."""

__version__ = "0.1.0"
```

`tests/test_import.py` :

```python
import tinyllm


def test_package_imports():
    assert tinyllm.__version__ == "0.1.0"
```

- [ ] **Étape 4 : Installer et vérifier**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Étape 5 : Commit**

```bash
git add pyproject.toml .gitignore src/tinyllm/__init__.py tests/test_import.py uv.lock
git commit -m "chore: bootstrap tinyllm package and tooling"
```

---
### Tâche 1 : Configurations typées

**Fichiers :**
- Créer : `src/tinyllm/config.py`
- Test : `tests/test_config.py`

- [ ] **Étape 1 : Écrire le test qui échoue**

```python
import json

import pytest

from tinyllm.config import ModelConfig, load_config


def test_head_dim_and_valid_config():
    cfg = ModelConfig(d_model=512, n_heads=8)
    assert cfg.head_dim == 64


def test_invalid_head_divisibility_raises():
    with pytest.raises(ValueError, match="d_model must be divisible"):
        ModelConfig(d_model=513, n_heads=8)


def test_invalid_kv_divisibility_raises():
    with pytest.raises(ValueError, match="n_heads must be divisible"):
        ModelConfig(n_heads=8, n_kv_heads=3)


def test_load_config_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_name": "smoke"}))
    cfg = load_config(path)
    assert cfg.run_name == "smoke"
    assert cfg.model.vocab_size == 16384
    assert cfg.train.micro_batch == 16
    assert cfg.data.tokenizer_path == "data/tokenizer.json"
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.config'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/config.py`**

```python
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 1408
    context_len: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TrainConfig:
    micro_batch: int = 16
    grad_accum: int = 8
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 7629
    eval_interval: int = 500
    checkpoint_interval: int = 500
    log_interval: int = 10
    sample_interval: int = 1000
    max_hours: float = 48.0
    seed: int = 42
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be 'float32' or 'bfloat16'")


@dataclass(frozen=True)
class DataConfig:
    tokenizer_path: str = "data/tokenizer.json"
    train_bin: str = "data/phase1/train.bin"
    val_bin: str = "data/phase1/val.bin"


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "phase1"
    output_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str | Path) -> RunConfig:
    raw = json.loads(Path(path).read_text())
    return RunConfig(
        run_name=raw["run_name"],
        output_dir=raw.get("output_dir", "runs"),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**raw.get("data", {})),
    )


def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2) + "\n")
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_config.py -q`
Expected: `4 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/config.py tests/test_config.py
git commit -m "feat: add typed run configuration"
```

---
### Tâche 2 : Tokenizer BPE byte-level

**Fichiers :**
- Créer : `src/tinyllm/tokenizer.py`
- Test : `tests/test_tokenizer.py`

- [ ] **Étape 1 : Écrire le test qui échoue**

```python
from pathlib import Path

from tinyllm.tokenizer import EOS_TOKEN, SPECIAL_TOKENS, TokenizerWrapper, train_tokenizer

CORPUS = [
    "Once upon a time there was a little girl named Lily.",
    "She found a small red ball in the garden and laughed.",
    "Le chat dort sur le tapis. Il rêve de poisson.",
    "def add(a, b):\n    return a + b\n",
] * 40


def _train(tmp_path: Path) -> TokenizerWrapper:
    path = tmp_path / "tokenizer.json"
    train_tokenizer(CORPUS, vocab_size=320, out_path=path)
    return TokenizerWrapper(path)


def test_roundtrip_is_lossless(tmp_path):
    tok = _train(tmp_path)
    text = "Hello, world! Ça va ? 123"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_exist(tmp_path):
    tok = _train(tmp_path)
    assert tok.eos_id == tok._tok.token_to_id(EOS_TOKEN)
    for token in SPECIAL_TOKENS:
        assert tok._tok.token_to_id(token) is not None


def test_token_bytes_matches_utf8_length(tmp_path):
    tok = _train(tmp_path)
    text = "Un garçon mange une pomme."
    total = sum(tok.token_bytes(i) for i in tok.encode(text))
    assert total == len(text.encode("utf-8"))


def test_save_and_reload(tmp_path):
    tok = _train(tmp_path)
    reloaded = TokenizerWrapper(Path(tok_path := tmp_path / "tokenizer.json"))
    assert reloaded.encode("hello") == tok.encode("hello")
    assert tok_path.exists()
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.tokenizer'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/tokenizer.py`**

```python
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"
SPECIAL_TOKENS = [EOS_TOKEN, "<|pad|>", "<|user|>", "<|assistant|>"]


def train_tokenizer(
    texts: Iterable[str], vocab_size: int = 16384, out_path: str | Path = "data/tokenizer.json"
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


class TokenizerWrapper:
    def __init__(self, path: str | Path) -> None:
        self._tok = Tokenizer.from_file(str(path))
        eos_id = self._tok.token_to_id(EOS_TOKEN)
        if eos_id is None:
            raise ValueError(f"tokenizer is missing {EOS_TOKEN}")
        self.eos_id = eos_id
        self._special_ids = {
            token_id
            for token in SPECIAL_TOKENS
            if (token_id := self._tok.token_to_id(token)) is not None
        }

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def token_bytes(self, token_id: int) -> int:
        if token_id in self._special_ids:
            return 0
        return len(self._tok.id_to_token(token_id))
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: `4 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add byte-level BPE tokenizer"
```

---
### Tâche 3 : Nettoyage, déduplication et splits

**Fichiers :**
- Créer : `src/tinyllm/data.py`
- Créer : `tests/test_data.py`

- [ ] **Étape 1 : Écrire le test qui échoue**

```python
from tinyllm.data import clean_text, dedupe, split_documents, text_hash


def test_clean_text_strips_html_and_whitespace():
    raw = "  <p>Hello</p>\n\n\n\n   world  \t !  "
    assert clean_text(raw) == "Hello\n\nworld !"


def test_dedupe_keeps_first_occurrence():
    docs = [{"text": "a"}, {"text": "b"}, {"text": "a"}]
    assert [d["text"] for d in dedupe(docs)] == ["a", "b"]


def test_split_is_deterministic_and_disjoint():
    docs = [{"text": f"document number {i} " + "x" * i} for i in range(2000)]
    first = split_documents(docs, val_frac=0.1, test_frac=0.1)
    second = split_documents(docs, val_frac=0.1, test_frac=0.1)
    assert [text_hash(d["text"]) for d in first["val"]] == [
        text_hash(d["text"]) for d in second["val"]
    ]
    assert 100 < len(first["val"]) < 300
    assert 100 < len(first["test"]) < 300
    ids = {name: {text_hash(d["text"]) for d in part} for name, part in first.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_data.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.data'`

- [ ] **Étape 3 : Implémenter la première moitié de `src/tinyllm/data.py`**

```python
import hashlib
import re
from collections.abc import Iterable, Iterator

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _TAG.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dedupe(docs: Iterable[dict], seen: set[str] | None = None) -> Iterator[dict]:
    seen = set() if seen is None else seen
    for doc in docs:
        digest = text_hash(doc["text"])
        if digest in seen:
            continue
        seen.add(digest)
        yield doc


def split_documents(
    docs: Iterable[dict], val_frac: float = 0.01, test_frac: float = 0.01
) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for doc in docs:
        position = int(text_hash(doc["text"])[:8], 16) / 0xFFFFFFFF
        if position < test_frac:
            splits["test"].append(doc)
        elif position < test_frac + val_frac:
            splits["val"].append(doc)
        else:
            splits["train"].append(doc)
    return splits
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_data.py -q`
Expected: `3 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/data.py tests/test_data.py
git commit -m "feat: add text cleaning, dedupe and deterministic splits"
```

---
### Tâche 4 : Packing et format binaire uint16

**Fichiers :**
- Modifier : `src/tinyllm/data.py`
- Modifier : `tests/test_data.py`

- [ ] **Étape 1 : Ajouter les tests qui échouent**

À la fin de `tests/test_data.py` :

```python
import numpy as np

from tinyllm.data import pack_documents, read_bin, write_bin


def test_pack_documents_inserts_eos_and_drops_remainder():
    docs = [[1, 2, 3], [4, 5]]
    assert list(pack_documents(docs, context_len=4, eos_id=9)) == [[1, 2, 3, 9]]
    assert list(pack_documents(docs, context_len=3, eos_id=9)) == [[1, 2, 3], [9, 4, 5]]


def test_pack_documents_all_chunks_have_context_len():
    docs = [[i] * 7 for i in range(5)]
    chunks = list(pack_documents(docs, context_len=8, eos_id=0))
    assert chunks
    assert all(len(chunk) == 8 for chunk in chunks)


def test_bin_roundtrip(tmp_path):
    path = tmp_path / "train.bin"
    written = write_bin([[1, 2, 3], [4, 5, 6]], path)
    assert written == 6
    array = read_bin(path)
    assert array.dtype == np.uint16
    assert array.tolist() == [1, 2, 3, 4, 5, 6]
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_data.py -q`
Expected: FAIL avec `ImportError: cannot import name 'pack_documents'`

- [ ] **Étape 3 : Ajouter l'implémentation à la fin de `src/tinyllm/data.py`**

Ajouter en tête du fichier :

```python
from pathlib import Path

import numpy as np
```

Puis à la fin du fichier :

```python
def pack_documents(
    documents: Iterable[list[int]], context_len: int, eos_id: int
) -> Iterator[list[int]]:
    buffer: list[int] = []
    for document in documents:
        buffer.extend(document)
        buffer.append(eos_id)
        while len(buffer) >= context_len:
            yield buffer[:context_len]
            buffer = buffer[context_len:]


def write_bin(chunks: Iterable[list[int]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as handle:
        for chunk in chunks:
            np.asarray(chunk, dtype=np.uint16).tofile(handle)
            written += len(chunk)
    return written


def read_bin(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size == 0:
        return np.empty(0, dtype=np.uint16)
    return np.memmap(path, dtype=np.uint16, mode="r")
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_data.py -q`
Expected: `6 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/data.py tests/test_data.py
git commit -m "feat: add document packing and uint16 binary format"
```

---
### Tâche 5 : CLI de préparation des données

**Fichiers :**
- Modifier : `src/tinyllm/data.py`
- Créer : `src/tinyllm/prepare.py`
- Modifier : `tests/test_data.py`
- Créer : `tests/test_prepare.py`

- [ ] **Étape 1 : Ajouter le test de `PackedWriter`**

À la fin de `tests/test_data.py` :

```python
from tinyllm.data import PackedWriter


def test_packed_writer_streams_complete_chunks(tmp_path):
    path = tmp_path / "out.bin"
    writer = PackedWriter(path, context_len=4, eos_id=9)
    writer.add_document([1, 2, 3])
    writer.add_document([4, 5, 6])
    written = writer.close()
    assert written == 4
    assert read_bin(path).tolist() == [1, 2, 3, 9]
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_data.py::test_packed_writer_streams_complete_chunks -q`
Expected: FAIL avec `ImportError: cannot import name 'PackedWriter'`

- [ ] **Étape 3 : Ajouter `PackedWriter` à `src/tinyllm/data.py`**

```python
class PackedWriter:
    def __init__(self, path: str | Path, context_len: int, eos_id: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context_len = context_len
        self.eos_id = eos_id
        self._buffer: list[int] = []
        self._handle = self.path.open("wb")
        self.tokens_written = 0

    def add_document(self, token_ids: list[int]) -> None:
        self._buffer.extend(token_ids)
        self._buffer.append(self.eos_id)
        self._flush()

    def _flush(self) -> None:
        while len(self._buffer) >= self.context_len:
            chunk = self._buffer[: self.context_len]
            np.asarray(chunk, dtype=np.uint16).tofile(self._handle)
            self.tokens_written += self.context_len
            self._buffer = self._buffer[self.context_len :]

    def close(self) -> int:
        self._handle.close()
        return self.tokens_written
```

- [ ] **Étape 4 : Lancer le test, vérifier le succès**

Run: `uv run pytest tests/test_data.py -q`
Expected: `7 passed`

- [ ] **Étape 5 : Ajouter le test d'intégration de `prepare_data`**

`tests/test_prepare.py` :

```python
import json
from pathlib import Path

from tinyllm.config import DataConfig, RunConfig
from tinyllm.prepare import prepare_data
from tinyllm.tokenizer import train_tokenizer

TEXTS = [f"Document {i}. " + "the quick brown fox jumps over the lazy dog " * (i % 5 + 1)
         for i in range(120)]


def _setup(tmp_path: Path) -> RunConfig:
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(TEXTS, vocab_size=320, out_path=tokenizer_path)
    return RunConfig(
        run_name="test",
        output_dir=str(tmp_path / "runs"),
        data=DataConfig(
            tokenizer_path=str(tokenizer_path),
            train_bin=str(tmp_path / "train.bin"),
            val_bin=str(tmp_path / "val.bin"),
            test_bin=str(tmp_path / "test.bin"),
        ),
    )


def _stream():
    for i, text in enumerate(TEXTS):
        yield "test", {"id": str(i), "text": text}


def test_prepare_data_writes_bins_and_meta(tmp_path):
    cfg = _setup(tmp_path)
    meta = prepare_data(cfg, document_stream=_stream, max_docs=120)
    assert meta["tokens"]["train"] > 0
    assert Path(cfg.data.train_bin).stat().st_size % 2 == 0
    on_disk = json.loads(Path(cfg.data.train_bin).with_name("meta.json").read_text())
    assert on_disk["context_len"] == cfg.model.context_len
    assert len(read_bin(cfg.data.train_bin)) % cfg.model.context_len == 0


def test_prepare_data_is_idempotent(tmp_path):
    cfg = _setup(tmp_path)
    first = prepare_data(cfg, document_stream=_stream, max_docs=120)
    second = prepare_data(cfg, document_stream=_stream, max_docs=120)
    assert first == second


from tinyllm.data import read_bin
```

- [ ] **Étape 6 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_prepare.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.prepare'`

- [ ] **Étape 7 : Implémenter `src/tinyllm/prepare.py`**

D'abord, ajouter à `src/tinyllm/config.py` ces champs à `DataConfig` :

```python
    sources: dict[str, float] = field(default_factory=lambda: {"tinystories": 1.0})
    val_frac: float = 0.01
    test_frac: float = 0.01
    min_chars: int = 200
    max_chars: int = 100_000
    test_bin: str = "data/phase1/test.bin"
```

Puis ajouter `split_for` à `src/tinyllm/data.py` et l'utiliser dans
`split_documents` (remplacer le corps de la boucle) :

```python
def split_for(text: str, val_frac: float, test_frac: float) -> str:
    position = int(text_hash(text)[:8], 16) / 0xFFFFFFFF
    if position < test_frac:
        return "test"
    if position < test_frac + val_frac:
        return "val"
    return "train"
```

Enfin créer `src/tinyllm/prepare.py` :

```python
import argparse
import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from tinyllm.config import RunConfig, load_config
from tinyllm.data import PackedWriter, clean_text, split_for, text_hash
from tinyllm.tokenizer import TokenizerWrapper

SOURCES = {
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "fineweb_edu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "wiki_en": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "wiki_fr": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.fr",
        "split": "train",
        "text_field": "text",
        "data_dir": None,
    },
    "python": {
        "name": "bigcode/the-stack-smol",
        "config": None,
        "split": "train",
        "text_field": "content",
        "data_dir": "data/python",
    },
}


def iter_hf_documents(source_name: str) -> Iterator[dict]:
    from datasets import load_dataset

    spec = SOURCES[source_name]
    dataset = load_dataset(
        spec["name"],
        spec["config"],
        split=spec["split"],
        streaming=True,
        data_dir=spec["data_dir"],
    )
    for row in dataset:
        text = row[spec["text_field"]]
        if text:
            yield {"text": text}


def iter_mixed_documents(
    mix: dict[str, float], seed: int, max_docs: int | None = None
) -> Iterator[tuple[str, dict]]:
    import random

    rng = random.Random(seed)
    names = list(mix)
    weights = [mix[name] for name in names]
    streams = {name: iter_hf_documents(name) for name in names}
    produced = 0
    while names and (max_docs is None or produced < max_docs):
        name = rng.choices(names, weights=weights, k=1)[0]
        try:
            document = next(streams[name])
        except StopIteration:
            index = names.index(name)
            names.pop(index)
            weights.pop(index)
            streams.pop(name)
            continue
        produced += 1
        yield name, document


def prepare_data(
    cfg: RunConfig,
    document_stream: Callable[[], Iterator[tuple[str, dict]]] | None = None,
    max_docs: int | None = None,
    force: bool = False,
) -> dict:
    train_path = Path(cfg.data.train_bin)
    meta_path = train_path.with_name("meta.json")
    if not force and train_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    if document_stream is None:
        document_stream = lambda: iter_mixed_documents(cfg.data.sources, cfg.train.seed)

    tokenizer = TokenizerWrapper(cfg.data.tokenizer_path)
    writers = {
        "train": PackedWriter(train_path, cfg.model.context_len, tokenizer.eos_id),
        "val": PackedWriter(cfg.data.val_bin, cfg.model.context_len, tokenizer.eos_id),
        "test": PackedWriter(cfg.data.test_bin, cfg.model.context_len, tokenizer.eos_id),
    }
    seen: set[str] = set()
    docs = 0
    tokens_by_source: dict[str, int] = {}
    for source_name, document in document_stream():
        text = clean_text(document["text"])
        if not (cfg.data.min_chars <= len(text) <= cfg.data.max_chars):
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        split = split_for(text, cfg.data.val_frac, cfg.data.test_frac)
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue
        writers[split].add_document(token_ids)
        tokens_by_source[source_name] = (
            tokens_by_source.get(source_name, 0) + len(token_ids) + 1
        )
        docs += 1
        if max_docs is not None and docs >= max_docs:
            break

    tokens = {name: writer.close() for name, writer in writers.items()}
    meta = {
        "sources": cfg.data.sources,
        "tokens": tokens,
        "tokens_by_source": tokens_by_source,
        "docs": docs,
        "context_len": cfg.model.context_len,
        "tokenizer_sha1": hashlib.sha1(
            Path(cfg.data.tokenizer_path).read_bytes()
        ).hexdigest(),
        "seed": cfg.train.seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    meta = prepare_data(cfg, max_docs=args.max_docs, force=args.force)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Étape 8 : Corriger l'import du test et lancer les tests**

Déplacer `from tinyllm.data import read_bin` en haut de `tests/test_prepare.py`,
puis :

Run: `uv run pytest tests/test_prepare.py tests/test_data.py -q`
Expected: `9 passed`

- [ ] **Étape 9 : Commit**

```bash
git add src/tinyllm/config.py src/tinyllm/data.py src/tinyllm/prepare.py tests/
git commit -m "feat: add data preparation CLI with mixed streaming sources"
```

---
### Tâche 6 : RMSNorm et RoPE

**Fichiers :**
- Créer : `src/tinyllm/model.py`
- Créer : `tests/test_model.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import mlx.core as mx

from tinyllm.model import RMSNorm, apply_rope, rope_tables


def test_rms_norm_has_unit_rms():
    norm = RMSNorm(64)
    x = mx.random.normal((2, 5, 64)) * 3.0
    out = norm(x)
    rms = mx.sqrt(mx.mean(out * out, axis=-1))
    assert mx.allclose(rms, mx.ones_like(rms), atol=1e-4)


def test_rope_at_position_zero_is_identity():
    x = mx.random.normal((1, 2, 1, 16))
    cos, sin = rope_tables(seq_len=4, head_dim=16, theta=10_000.0)
    assert mx.allclose(apply_rope(x, cos, sin, offset=0), x, atol=1e-6)


def test_rope_preserves_norm():
    x = mx.random.normal((1, 2, 5, 16))
    cos, sin = rope_tables(seq_len=8, head_dim=16, theta=10_000.0)
    out = apply_rope(x, cos, sin, offset=0)
    assert mx.allclose(
        mx.linalg.norm(x, axis=-1), mx.linalg.norm(out, axis=-1), atol=1e-4
    )


def test_rope_depends_only_on_relative_position():
    q = mx.random.normal((1, 1, 1, 16))
    k = mx.random.normal((1, 1, 1, 16))
    cos, sin = rope_tables(seq_len=12, head_dim=16, theta=10_000.0)
    q3 = apply_rope(q, cos, sin, offset=3)
    k5 = apply_rope(k, cos, sin, offset=5)
    q0 = apply_rope(q, cos, sin, offset=0)
    k2 = apply_rope(k, cos, sin, offset=2)
    assert mx.allclose(mx.sum(q3 * k5), mx.sum(q0 * k2), atol=1e-4)
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.model'`

- [ ] **Étape 3 : Implémenter les primitives dans `src/tinyllm/model.py`**

```python
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def rope_tables(
    seq_len: int, head_dim: int, theta: float, dtype=mx.float32
) -> tuple[mx.array, mx.array]:
    half = head_dim // 2
    inv_freq = theta ** (-mx.arange(0, half, dtype=mx.float32) / half)
    positions = mx.arange(seq_len, dtype=mx.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    return cos.astype(dtype), sin.astype(dtype)


def apply_rope(
    x: mx.array, cos: mx.array, sin: mx.array, offset: int = 0
) -> mx.array:
    seq_len = x.shape[2]
    cos = cos[offset : offset + seq_len][None, None, :, :]
    sin = sin[offset : offset + seq_len][None, None, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_model.py -q`
Expected: `4 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/model.py tests/test_model.py
git commit -m "feat: add RMSNorm and rotary position embeddings"
```

---
### Tâche 7 : Attention causale GQA

**Fichiers :**
- Modifier : `src/tinyllm/model.py`
- Modifier : `tests/test_model.py`

- [ ] **Étape 1 : Ajouter les tests qui échouent**

À la fin de `tests/test_model.py` :

```python
import numpy as np

from tinyllm.config import ModelConfig
from tinyllm.model import Attention, causal_mask


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=320,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        context_len=32,
    )


def test_causal_mask_lower_triangle():
    mask = np.array(causal_mask(3, 3))
    assert np.all(np.isfinite(mask[np.tril_indices(3)]))
    assert np.all(np.isneginf(mask[np.triu_indices(3, k=1)]))


def test_causal_mask_single_query_sees_everything():
    assert np.all(np.isfinite(np.array(causal_mask(1, 5))))


def test_attention_output_shape():
    cfg = _cfg()
    attention = Attention(cfg)
    x = mx.random.normal((2, 16, cfg.d_model))
    cos, sin = rope_tables(16, cfg.head_dim, cfg.rope_theta)
    out = attention(x, cos, sin)
    assert out.shape == (2, 16, cfg.d_model)


def test_attention_is_causal():
    cfg = _cfg()
    attention = Attention(cfg)
    mx.random.seed(0)
    x = mx.random.normal((1, 10, cfg.d_model))
    cos, sin = rope_tables(10, cfg.head_dim, cfg.rope_theta)
    out_a = attention(x, cos, sin)
    mask = (mx.arange(10) == 5)[None, :, None]
    changed = mx.where(mask, x + 1.0, x)
    out_b = attention(changed, cos, sin)
    assert mx.allclose(out_a[:, :5], out_b[:, :5], atol=1e-5)
    assert not mx.allclose(out_a[:, 5:], out_b[:, 5:], atol=1e-5)
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL avec `ImportError: cannot import name 'Attention'`

- [ ] **Étape 3 : Ajouter l'implémentation à `src/tinyllm/model.py`**

Ajouter en tête du fichier :

```python
from tinyllm.config import ModelConfig
```

Puis à la fin du fichier :

```python
def causal_mask(q_len: int, kv_len: int, dtype=mx.float32) -> mx.array:
    q_pos = mx.arange(q_len)[:, None] + (kv_len - q_len)
    k_pos = mx.arange(kv_len)[None, :]
    return mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim**-0.5
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cache=None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = 0 if cache is None else cache.length
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)
        if cache is not None:
            k, v = cache.update(k, v)

        repeats = self.n_heads // self.n_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + causal_mask(seq_len, k.shape[2], scores.dtype)
        probs = mx.softmax(scores, axis=-1)
        out = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_model.py -q`
Expected: `8 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/model.py tests/test_model.py
git commit -m "feat: add causal grouped-query attention"
```

---
### Tâche 8 : MLP, bloc et assemblage de TinyLM

**Fichiers :**
- Modifier : `src/tinyllm/model.py`
- Modifier : `tests/test_model.py`

- [ ] **Étape 1 : Ajouter les tests qui échouent**

À la fin de `tests/test_model.py` :

```python
from mlx.utils import tree_flatten

from tinyllm.model import MLP, Block, TinyLM


def test_mlp_shape():
    cfg = _cfg()
    mlp = MLP(cfg)
    assert mlp(mx.random.normal((2, 7, cfg.d_model))).shape == (2, 7, cfg.d_model)


def test_block_shape():
    cfg = _cfg()
    block = Block(cfg)
    x = mx.random.normal((2, 7, cfg.d_model))
    cos, sin = rope_tables(7, cfg.head_dim, cfg.rope_theta)
    assert block(x, cos, sin).shape == (2, 7, cfg.d_model)


def test_tinylm_logits_shape_and_dtype():
    cfg = _cfg()
    model = TinyLM(cfg)
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert logits.dtype == mx.float32


def test_parameter_count_matches_formula():
    cfg = _cfg()
    model = TinyLM(cfg)
    attention = 2 * cfg.d_model**2 + 2 * cfg.d_model * cfg.n_kv_heads * cfg.head_dim
    mlp = 3 * cfg.d_model * cfg.ffn_hidden
    per_block = attention + mlp + 2 * cfg.d_model
    expected = (
        cfg.vocab_size * cfg.d_model
        + cfg.n_layers * per_block
        + cfg.d_model
    )
    assert model.num_parameters() == expected


def test_embeddings_are_tied():
    model = TinyLM(_cfg())
    keys = [str(key) for key, _ in tree_flatten(model.parameters())]
    assert not any(key.startswith("lm_head") for key in keys)
    assert any(key.startswith("tok_emb") for key in keys)


def test_tinylm_is_causal():
    cfg = _cfg()
    model = TinyLM(cfg)
    mx.random.seed(1)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits_a = model(tokens)
    mask = (mx.arange(8) == 4)[None, :, None]
    replacement = mx.full(tokens.shape, 300, dtype=tokens.dtype)
    modified = mx.where(mask, replacement, tokens)
    logits_b = model(modified)
    assert mx.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5)
    assert not mx.allclose(logits_a[:, 4:], logits_b[:, 4:], atol=1e-5)
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_model.py -q`
Expected: FAIL avec `ImportError: cannot import name 'MLP'`

- [ ] **Étape 3 : Ajouter `MLP`, `Block` et `TinyLM` à `src/tinyllm/model.py`**

```python
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = MLP(cfg)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, cache=None
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.ffn(self.ffn_norm(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def __call__(self, tokens: mx.array, cache=None) -> mx.array:
        x = self.tok_emb(tokens)
        cos, sin = rope_tables(
            self.cfg.context_len,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            dtype=x.dtype,
        )
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            x = layer(x, cos, sin, layer_cache)
        x = self.norm(x)
        return (x @ self.tok_emb.weight.T).astype(mx.float32)

    def num_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(parameter.size for _, parameter in tree_flatten(self.parameters()))
```

Le stockage de `cos`/`sin` est volontairement recalculé à chaque forward :
dans MLX, tout `mx.array` stocké comme attribut entre dans `parameters()`.
Les recalculer coûte moins de 1 % du temps d'un step.

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_model.py -q`
Expected: `14 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/model.py tests/test_model.py
git commit -m "feat: assemble TinyLM with tied embeddings"
```

---
### Tâche 9 : Cache KV

**Fichiers :**
- Modifier : `src/tinyllm/model.py`
- Modifier : `tests/test_model.py`

- [ ] **Étape 1 : Ajouter les tests qui échouent**

À la fin de `tests/test_model.py` :

```python
from tinyllm.model import KVCache


def test_cache_length_starts_at_zero():
    assert KVCache().length == 0


def test_kv_cache_matches_full_forward():
    cfg = _cfg()
    model = TinyLM(cfg)
    mx.random.seed(3)
    tokens = mx.random.randint(0, cfg.vocab_size, (1, 12))
    full = model(tokens)

    cache = model.make_cache()
    prefill = model(tokens[:, :8], cache=cache)
    assert cache[0].length == 8
    assert mx.allclose(prefill, full[:, :8], atol=1e-4)

    steps = []
    for position in range(8, 12):
        logits = model(tokens[:, position : position + 1], cache=cache)
        steps.append(logits[:, 0, :])
    incremental = mx.stack(steps, axis=1)
    assert cache[0].length == 12
    assert mx.allclose(incremental, full[:, 8:, :], atol=1e-4)
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_model.py::test_kv_cache_matches_full_forward -q`
Expected: FAIL avec `ImportError: cannot import name 'KVCache'`

- [ ] **Étape 3 : Ajouter `KVCache` et `make_cache`**

Dans `src/tinyllm/model.py`, avant `Attention` :

```python
class KVCache:
    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values
```

Puis ajouter cette méthode à `TinyLM` :

```python
    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_model.py -q`
Expected: `16 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/model.py tests/test_model.py
git commit -m "feat: add incremental KV cache"
```

---
### Tâche 10 : Loss, précision mixte et accumulation de gradient

**Fichiers :**
- Créer : `src/tinyllm/train.py`
- Créer : `tests/test_train.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.train import (
    cast_tree,
    clip_grads,
    grad_norm,
    lr_at_step,
    loss_fn,
    make_optimizer,
)


def _cfg() -> tuple[ModelConfig, TrainConfig]:
    model = ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, context_len=16,
    )
    train = TrainConfig(lr=1e-2, max_steps=100, warmup_ratio=0.1, micro_batch=2)
    return model, train


def test_loss_is_scalar_and_near_ln_vocab_at_init():
    model_cfg, _ = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    tokens = mx.random.randint(0, model_cfg.vocab_size, (2, 9))
    loss = loss_fn(model, tokens)
    assert loss.shape == ()
    assert abs(float(loss) - math.log(model_cfg.vocab_size)) < 0.5


def test_grad_norm_of_known_tree():
    grads = {"a": mx.array([3.0, 4.0]), "b": mx.array([12.0])}
    assert mx.allclose(grad_norm(grads), mx.array(13.0), atol=1e-5)


def test_clip_grads_caps_norm():
    grads = {"w": mx.array([3.0, 4.0])}
    clipped, norm = clip_grads(grads, max_norm=1.0)
    assert mx.allclose(norm, mx.array(5.0), atol=1e-5)
    assert mx.allclose(grad_norm(clipped), mx.array(1.0), atol=1e-4)


def test_lr_schedule_endpoints():
    _, train_cfg = _cfg()
    warmup = int(train_cfg.warmup_ratio * train_cfg.max_steps)
    assert abs(lr_at_step(0, train_cfg) - train_cfg.lr / warmup) < 1e-9
    assert abs(lr_at_step(train_cfg.max_steps - 1, train_cfg) - train_cfg.lr * 0.1) < 1e-5


def test_one_step_reduces_loss_on_single_batch():
    model_cfg, train_cfg = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    work = TinyLM(model_cfg)
    optimizer = make_optimizer(train_cfg)
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    batch = mx.random.randint(0, model_cfg.vocab_size, (2, 9))

    initial = float(loss_fn(model, batch))
    for _ in range(20):
        work.update(cast_tree(model.parameters(), mx.bfloat16))
        loss, grads = loss_and_grad(work, batch)
        grads = cast_tree(grads, mx.float32)
        optimizer.update(model, grads)
    final = float(loss_fn(model, batch))
    assert final < initial * 0.9


def test_accumulated_grads_match_single_batch():
    model_cfg, _ = _cfg()
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    work = TinyLM(model_cfg)
    work.update(cast_tree(model.parameters(), mx.float32))
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    batch = mx.random.randint(0, model_cfg.vocab_size, (4, 9))

    _, full_grads = loss_and_grad(work, batch)
    _, grad_a = loss_and_grad(work, batch[:2])
    _, grad_b = loss_and_grad(work, batch[2:])
    averaged = tree_map(lambda a, b: (a + b) / 2, grad_a, grad_b)
    for (_, g_full), (_, g_avg) in zip(tree_flatten(full_grads), tree_flatten(averaged)):
        assert mx.allclose(g_full, g_avg, atol=1e-4)
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_train.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.train'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/train.py` (partie mathématique)**

```python
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tinyllm.config import TrainConfig


def loss_fn(model: nn.Module, tokens: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def grad_norm(grads) -> mx.array:
    total = mx.zeros(())
    for _, gradient in tree_flatten(grads):
        total = total + mx.sum(gradient.astype(mx.float32) ** 2)
    return mx.sqrt(total)


def clip_grads(grads, max_norm: float):
    norm = grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm


def cast_tree(tree, dtype):
    return tree_map(lambda value: value.astype(dtype), tree)


def make_optimizer(cfg: TrainConfig) -> optim.AdamW:
    return optim.AdamW(
        learning_rate=cfg.lr,
        betas=[cfg.beta1, cfg.beta2],
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + (cfg.lr - min_lr) * cosine
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_train.py -q`
Expected: `6 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/train.py tests/test_train.py
git commit -m "feat: add loss, gradient clipping and lr schedule"
```

---
### Tâche 11 : Checkpoints et reprise exacte

**Fichiers :**
- Créer : `src/tinyllm/checkpoint.py`
- Créer : `tests/test_checkpoint.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from tinyllm.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.train import make_optimizer


def _setup():
    model_cfg = ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, context_len=16,
    )
    train_cfg = TrainConfig(lr=1e-2, max_steps=10, warmup_ratio=0.1)
    mx.random.seed(0)
    model = TinyLM(model_cfg)
    optimizer = make_optimizer(train_cfg)
    return model, optimizer, train_cfg


def _train_step(model, work, optimizer, batch):
    loss_and_grad = nn.value_and_grad(work, lambda m, t: nn.losses.cross_entropy(
        m(t[:, :-1]), t[:, 1:], reduction="mean"
    ))
    work.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    _, grads = loss_and_grad(work, batch)
    optimizer.update(model, grads)


def test_save_then_load_restores_parameters_and_meta(tmp_path):
    model, optimizer, train_cfg = _setup()
    batch = mx.random.randint(0, 320, (2, 9))
    _train_step(model, TinyLM(model.cfg), optimizer, batch)
    before = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    save_checkpoint(tmp_path, model, optimizer, {"step": 1, "best_val": 2.5})
    model.update(tree_map(lambda value: value * 0.0, model.parameters()))
    meta = load_checkpoint(tmp_path, model, optimizer)

    after = dict(tree_flatten(model.parameters()))
    assert meta["step"] == 1
    assert all(mx.allclose(before[k], after[k]) for k in before)


def test_resume_reproduces_continuation(tmp_path):
    model_a, optimizer_a, _ = _setup()
    batch = mx.random.randint(0, 320, (2, 9))
    work_a = TinyLM(model_a.cfg)
    _train_step(model_a, work_a, optimizer_a, batch)
    _train_step(model_a, work_a, optimizer_a, batch)

    model_b, optimizer_b, _ = _setup()
    work_b = TinyLM(model_b.cfg)
    _train_step(model_b, work_b, optimizer_b, batch)
    save_checkpoint(tmp_path, model_b, optimizer_b, {"step": 1})

    model_c, optimizer_c, _ = _setup()
    load_checkpoint(tmp_path, model_c, optimizer_c)
    _train_step(model_c, TinyLM(model_c.cfg), optimizer_c, batch)

    params_a = dict(tree_flatten(model_a.parameters()))
    params_c = dict(tree_flatten(model_c.parameters()))
    assert all(mx.allclose(params_a[k], params_c[k], atol=1e-6) for k in params_a)


def test_load_missing_checkpoint_raises(tmp_path):
    model, optimizer, _ = _setup()
    try:
        load_checkpoint(tmp_path / "missing", model, optimizer)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_checkpoint.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.checkpoint'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/checkpoint.py`**

```python
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
        optimizer.state = tree_unflatten(
            mx.load(str(path / "optimizer.safetensors"))
        )
    return json.loads((path / "state.json").read_text())
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_checkpoint.py -q`
Expected: `3 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add resumable checkpoints with optimizer state"
```

---
### Tâche 12 : Dataset et batching

**Fichiers :**
- Créer : `src/tinyllm/dataset.py`
- Créer : `tests/test_dataset.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import numpy as np

from tinyllm.data import write_bin
from tinyllm.dataset import TokenDataset


def _make_bin(tmp_path, n_tokens=1000):
    path = tmp_path / "train.bin"
    write_bin([list(range(i, i + 50)) for i in range(0, n_tokens, 50)], path)
    return path


def test_batch_shape_and_dtype(tmp_path):
    data = TokenDataset(_make_bin(tmp_path), context_len=16, seed=0)
    batch = data.next_batch(4)
    assert batch.shape == (4, 17)
    assert str(batch.dtype) == "mlx.core.int32"


def test_same_seed_same_order(tmp_path):
    path = _make_bin(tmp_path)
    a = TokenDataset(path, context_len=16, seed=7)
    b = TokenDataset(path, context_len=16, seed=7)
    assert np.array_equal(np.array(a.next_batch(8)), np.array(b.next_batch(8)))


def test_state_roundtrip_resumes_identical_stream(tmp_path):
    path = _make_bin(tmp_path)
    dataset = TokenDataset(path, context_len=16, seed=3)
    dataset.next_batch(8)
    state = dataset.state()

    resumed = TokenDataset(path, context_len=16, seed=3)
    resumed.load_state(state)
    expected = np.array(dataset.next_batch(8))
    actual = np.array(resumed.next_batch(8))
    assert np.array_equal(expected, actual)


def test_epoch_advances_after_full_pass(tmp_path):
    data = TokenDataset(_make_bin(tmp_path, n_tokens=200), context_len=16, seed=0)
    batches = 0
    while data.epoch == 0 and batches < 100:
        data.next_batch(8)
        batches += 1
    assert data.epoch == 1
    assert batches >= 1
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.dataset'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/dataset.py`**

```python
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
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_dataset.py -q`
Expected: `4 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/dataset.py tests/test_dataset.py
git commit -m "feat: add memmap dataset with resumable shuffling"
```

---
### Tâche 13 : Boucle d'entraînement

**Fichiers :**
- Modifier : `src/tinyllm/train.py`
- Créer : `tests/test_train_loop.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from tinyllm.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from tinyllm.data import write_bin
from tinyllm.train import rotate_checkpoints, train


def _cfg(tmp_path: Path, name: str, max_steps: int = 6) -> RunConfig:
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    if not train_bin.exists():
        write_bin(
            [[(i + j) % 300 for j in range(40)] for i in range(0, 2000, 40)],
            train_bin,
        )
        write_bin(
            [[(i + j) % 300 for j in range(40)] for i in range(2000, 2400, 40)],
            val_bin,
        )
    return RunConfig(
        run_name=name,
        output_dir=str(tmp_path / "runs"),
        model=ModelConfig(
            vocab_size=320, d_model=32, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_hidden=64, context_len=16,
        ),
        train=TrainConfig(
            micro_batch=2, grad_accum=2, lr=1e-2, max_steps=max_steps,
            eval_interval=3, checkpoint_interval=3, log_interval=1,
            dtype="float32", max_hours=1,
        ),
        data=DataConfig(
            tokenizer_path="", train_bin=str(train_bin), val_bin=str(val_bin)
        ),
    )


def test_train_writes_checkpoint_and_logs(tmp_path):
    cfg = _cfg(tmp_path, "smoke")
    train(cfg)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    meta = json.loads((run_dir / "checkpoints/latest/state.json").read_text())
    assert meta["step"] == 6
    records = [
        json.loads(line)
        for line in (run_dir / "log.jsonl").read_text().splitlines()
    ]
    assert len(records) == 6
    assert all(np.isfinite(record["loss"]) for record in records)
    assert (run_dir / "config.json").exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    train(_cfg(tmp_path, "full", max_steps=6))
    train(_cfg(tmp_path, "part", max_steps=3))
    train(
        _cfg(tmp_path, "resumed", max_steps=6),
        resume=tmp_path / "runs/part/checkpoints/latest",
    )
    full = mx.load(str(tmp_path / "runs/full/checkpoints/latest/model.safetensors"))
    resumed = mx.load(str(tmp_path / "runs/resumed/checkpoints/latest/model.safetensors"))
    assert full.keys() == resumed.keys()
    assert all(mx.allclose(full[key], resumed[key], atol=1e-5) for key in full)


def test_rotate_checkpoints_keeps_last_and_best(tmp_path):
    for step in (1, 2, 3):
        (tmp_path / f"step_{step:07d}").mkdir()
    (tmp_path / "best").mkdir()
    rotate_checkpoints(tmp_path, keep=2)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["best", "step_0000002", "step_0000003"]
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_train_loop.py -q`
Expected: FAIL avec `ImportError: cannot import name 'train'`

- [ ] **Étape 3 : Ajouter imports et utilitaires à `src/tinyllm/train.py`**

En tête du fichier :

```python
import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from tinyllm.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.config import RunConfig, TrainConfig, load_config, save_config
from tinyllm.dataset import TokenDataset
from tinyllm.model import TinyLM
```

À la fin du fichier :

```python
def evaluate_loss(
    model, path: str, context_len: int, batch_size: int, batches: int = 10
) -> float:
    dataset = TokenDataset(path, context_len, seed=0)
    if len(dataset.data) <= context_len + 1:
        return float("nan")
    losses = [float(loss_fn(model, dataset.next_batch(batch_size))) for _ in range(batches)]
    return sum(losses) / len(losses)


def rotate_checkpoints(ckpt_dir: str | Path, keep: int = 3) -> None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return
    step_dirs = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in step_dirs[:-keep] if keep > 0 else step_dirs:
        shutil.rmtree(old)
```

- [ ] **Étape 4 : Ajouter `train` à `src/tinyllm/train.py`**

```python
def train(
    cfg: RunConfig,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    force: bool = False,
) -> dict:
    if max_steps is not None:
        cfg = replace(cfg, train=replace(cfg.train, max_steps=max_steps))

    run_dir = Path(cfg.output_dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    dtype = mx.bfloat16 if cfg.train.dtype == "bfloat16" else mx.float32
    model = TinyLM(cfg.model)
    work = TinyLM(cfg.model)
    optimizer = make_optimizer(cfg.train)
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    dataset = TokenDataset(cfg.data.train_bin, cfg.model.context_len, cfg.train.seed)

    step = 0
    best_val = float("inf")
    if resume is not None:
        meta = load_checkpoint(resume, model, optimizer)
        step = int(meta["step"])
        best_val = float(meta.get("best_val", best_val))
        dataset.load_state(meta["dataset"])

    log_path = run_dir / "log.jsonl"
    tokens_per_step = (
        cfg.train.micro_batch * cfg.train.grad_accum * cfg.model.context_len
    )
    start = time.perf_counter()

    while step < cfg.train.max_steps:
        learning_rate = lr_at_step(step, cfg.train)
        optimizer.learning_rate = learning_rate
        accumulated = None
        for _ in range(cfg.train.grad_accum):
            batch = dataset.next_batch(cfg.train.micro_batch)
            work.update(cast_tree(model.parameters(), dtype))
            loss, grads = loss_and_grad(work, batch)
            grads = cast_tree(grads, mx.float32)
            accumulated = (
                grads
                if accumulated is None
                else tree_map(lambda a, b: a + b, accumulated, grads)
            )
        accumulated = tree_map(lambda g: g / cfg.train.grad_accum, accumulated)
        accumulated, norm = clip_grads(accumulated, cfg.train.grad_clip)
        optimizer.update(model, accumulated)
        mx.eval(model.parameters(), optimizer.state)
        step += 1
        elapsed = time.perf_counter() - start

        if step % cfg.train.log_interval == 0:
            record = {
                "step": step,
                "tokens": step * tokens_per_step,
                "loss": float(loss),
                "lr": learning_rate,
                "grad_norm": float(norm),
                "tokens_per_s": step * tokens_per_step / elapsed,
                "peak_memory_mb": mx.metal.get_peak_memory() / 2**20,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        if step == 100 and not force and step < cfg.train.max_steps:
            estimated_hours = elapsed / step * cfg.train.max_steps / 3600
            if estimated_hours > cfg.train.max_hours:
                raise SystemExit(
                    f"estimated {estimated_hours:.1f}h > max_hours="
                    f"{cfg.train.max_hours}; use --force to override"
                )

        val_loss = None
        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
            val_loss = evaluate_loss(
                model,
                cfg.data.val_bin,
                cfg.model.context_len,
                cfg.train.micro_batch,
            )

        if step % cfg.train.checkpoint_interval == 0 or step == cfg.train.max_steps:
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
            meta = {
                "step": step,
                "best_val": best_val,
                "dataset": dataset.state(),
                "elapsed_s": elapsed,
            }
            save_checkpoint(ckpt_dir / f"step_{step:07d}", model, optimizer, meta)
            save_checkpoint(ckpt_dir / "latest", model, optimizer, meta)
            if val_loss is not None and val_loss <= best_val:
                save_checkpoint(ckpt_dir / "best", model, optimizer, meta)
            rotate_checkpoints(ckpt_dir, keep=3)

    return {"step": step, "best_val": best_val}
```

- [ ] **Étape 5 : Ajouter l'arrêt propre et le garde-fou mémoire**

Ajouter `import signal` aux imports de `src/tinyllm/train.py`, puis avant `def train` :

```python
_STOP = {"requested": False}


def _request_stop(signum, frame) -> None:
    _STOP["requested"] = True
```

Dans `train()`, juste avant la boucle `while` :

```python
    signal.signal(signal.SIGINT, _request_stop)
    _STOP["requested"] = False
```

À la fin de chaque itération, après le bloc des checkpoints :

```python
        peak_mb = mx.metal.get_peak_memory() / 2**20
        if peak_mb > 13_000:
            raise SystemExit(
                f"peak memory {peak_mb:.0f} MB > 13000 MB; "
                "reduce micro_batch and increase grad_accum"
            )
        if _STOP["requested"]:
            print("interrupt received, saving checkpoint")
            break
```

Après la boucle `while`, avant le `return` :

```python
    meta = {
        "step": step,
        "best_val": best_val,
        "dataset": dataset.state(),
        "elapsed_s": time.perf_counter() - start,
    }
    save_checkpoint(ckpt_dir / "latest", model, optimizer, meta)
```

- [ ] **Étape 6 : Ajouter le CLI à la fin de `src/tinyllm/train.py`**

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = train(
        load_config(args.config),
        resume=args.resume,
        max_steps=args.max_steps,
        force=args.force,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
```

- [ ] **Étape 7 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_train_loop.py -q`
Expected: `3 passed`

- [ ] **Étape 8 : Commit**

```bash
git add src/tinyllm/train.py tests/test_train_loop.py
git commit -m "feat: add training loop with resume and guardrails"
```

---
### Tâche 14 : Génération et sampling

**Fichiers :**
- Créer : `src/tinyllm/sample.py`
- Créer : `tests/test_sample.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import mlx.core as mx
import numpy as np

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.sample import (
    apply_repetition_penalty,
    filter_logits,
    generate_ids,
    sample_token,
)


def _model() -> TinyLM:
    mx.random.seed(0)
    return TinyLM(
        ModelConfig(
            vocab_size=320, d_model=32, n_layers=1, n_heads=4,
            n_kv_heads=2, ffn_hidden=64, context_len=16,
        )
    )


def test_generate_ids_length_without_eos():
    model = _model()
    ids = generate_ids(
        model, [1, 2, 3], max_tokens=5, temperature=0.8, stop_at_eos=False, seed=0
    )
    assert len(ids) == 8


def test_greedy_generation_is_deterministic():
    model = _model()
    a = generate_ids(model, [1, 2], max_tokens=6, temperature=0.0, stop_at_eos=False)
    b = generate_ids(model, [1, 2], max_tokens=6, temperature=0.0, stop_at_eos=False)
    assert a == b


def test_top_k_keeps_single_token():
    logits = mx.array([1.0, 4.0, 2.0, 3.0])
    filtered = filter_logits(logits, top_k=1)
    assert int(mx.argmax(filtered)) == 1
    assert np.isneginf(np.array(filtered)).sum() == 3


def test_top_p_keeps_at_least_one_token():
    logits = mx.array([10.0, 0.0, 0.0])
    filtered = filter_logits(logits, top_p=0.1)
    assert np.isfinite(np.array(filtered)).sum() >= 1


def test_repetition_penalty_changes_seen_token():
    logits = mx.array([0.0, 0.0, 3.0, 0.0])
    penalized = apply_repetition_penalty(logits, [2, 2], penalty=2.0)
    assert abs(float(penalized[2]) - 1.5) < 1e-6


def test_sample_token_greedy_matches_argmax():
    logits = mx.array([0.0, 5.0, 1.0])
    key = mx.random.key(0)
    assert sample_token(logits, temperature=0.0, key=key) == 1
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_sample.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.sample'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/sample.py`**

```python
import argparse
import json
from pathlib import Path

import mlx.core as mx

from tinyllm.config import RunConfig, load_config
from tinyllm.model import TinyLM
from tinyllm.tokenizer import TokenizerWrapper


def apply_repetition_penalty(
    logits: mx.array, token_ids: list[int], penalty: float
) -> mx.array:
    if penalty <= 1.0 or not token_ids:
        return logits
    ids = mx.array(sorted(set(token_ids)))
    selected = logits[ids]
    penalized = mx.where(selected > 0, selected / penalty, selected * penalty)
    return logits.at[ids].add(penalized - selected)


def filter_logits(
    logits: mx.array, top_k: int | None = None, top_p: float | None = None
) -> mx.array:
    if top_k is not None and top_k > 0:
        keep = min(top_k, logits.shape[-1])
        cutoff = mx.sort(logits)[-keep]
        logits = mx.where(logits < cutoff, -mx.inf, logits)
    if top_p is not None and top_p < 1.0:
        order = mx.argsort(logits)[::-1]
        ordered = logits[order]
        probabilities = mx.softmax(ordered)
        cumulative = mx.cumsum(probabilities)
        drop = cumulative - probabilities > top_p
        ordered = mx.where(drop, -mx.inf, ordered)
        logits = mx.full(logits.shape, -mx.inf).at[order].add(ordered)
    return logits


def sample_token(
    logits: mx.array,
    temperature: float,
    key: mx.array,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    if temperature <= 0:
        return int(mx.argmax(logits))
    logits = filter_logits(logits / temperature, top_k, top_p)
    return int(mx.random.categorical(logits, key=key))


def generate_ids(
    model: TinyLM,
    prompt_ids: list[int],
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.0,
    stop_at_eos: bool = True,
    eos_id: int | None = None,
    seed: int = 0,
) -> list[int]:
    key = mx.random.key(seed)
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)[0, -1, :]
    generated: list[int] = []
    for _ in range(max_tokens):
        logits = apply_repetition_penalty(
            logits, prompt_ids + generated, repetition_penalty
        )
        key, subkey = mx.random.split(key)
        token = sample_token(logits, temperature, subkey, top_k, top_p)
        if stop_at_eos and eos_id is not None and token == eos_id:
            break
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)[0, -1, :]
    return prompt_ids + generated


def generate(
    model: TinyLM,
    tokenizer: TokenizerWrapper,
    prompt: str,
    **kwargs,
) -> str:
    return tokenizer.decode(
        generate_ids(
            model,
            tokenizer.encode(prompt),
            eos_id=tokenizer.eos_id,
            **kwargs,
        )
    )


def load_run_model(
    run_dir: str | Path, checkpoint: str = "latest"
) -> tuple[TinyLM, TokenizerWrapper, RunConfig]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    model = TinyLM(cfg.model)
    weights = mx.load(str(run_dir / "checkpoints" / checkpoint / "model.safetensors"))
    model.load_weights(list(weights.items()))
    return model, TokenizerWrapper(cfg.data.tokenizer_path), cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model, tokenizer, _ = load_run_model(args.run_dir, args.checkpoint)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_sample.py -q`
Expected: `6 passed`

- [ ] **Étape 5 : Brancher les échantillons périodiques dans `train.py`**

Ajouter en tête de `src/tinyllm/train.py` :

```python
PROMPTS = ["Once upon a time", "The little girl", "One day, the cat"]
```

Dans `train()`, après la création de `log_path` :

```python
    tokenizer = None
    tokenizer_path = Path(cfg.data.tokenizer_path)
    if tokenizer_path.exists():
        from tinyllm.sample import generate
        from tinyllm.tokenizer import TokenizerWrapper

        tokenizer = TokenizerWrapper(tokenizer_path)
```

Toujours dans `train()`, après le bloc de logging :

```python
        if tokenizer is not None and step % cfg.train.sample_interval == 0:
            samples_dir = run_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            blocks = []
            for prompt in PROMPTS:
                text = generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=120,
                    temperature=0.8,
                    top_p=0.9,
                    seed=step,
                )
                blocks.append(f"--- {prompt}\n{text}\n")
            (samples_dir / f"step_{step:07d}.txt").write_text("\n".join(blocks))
```

- [ ] **Étape 6 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_sample.py tests/test_train_loop.py -q`
Expected: `9 passed`

- [ ] **Étape 7 : Commit**

```bash
git add src/tinyllm/sample.py tests/test_sample.py
git commit -m "feat: add sampling with kv cache and text generation cli"
```

---
### Tâche 15 : Évaluation bits-per-byte

**Fichiers :**
- Créer : `src/tinyllm/eval.py`
- Créer : `tests/test_eval.py`

- [ ] **Étape 1 : Écrire les tests qui échouent**

```python
import math

import mlx.core as mx
import mlx.nn as nn

from tinyllm.data import write_bin
from tinyllm.eval import evaluate_bpb


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, tokens: mx.array) -> mx.array:
        return mx.zeros((*tokens.shape, self.vocab_size))


class FakeTokenizer:
    def token_bytes(self, token_id: int) -> int:
        return 1


def test_uniform_model_bpb_equals_log2_vocab(tmp_path):
    vocab = 64
    path = tmp_path / "val.bin"
    write_bin([list(range(9))], path)
    result = evaluate_bpb(
        UniformModel(vocab), path, FakeTokenizer(), context_len=4, batch_size=2
    )
    assert abs(result["bpb"] - math.log2(vocab)) < 1e-4
    assert result["tokens"] == 8
    assert result["bytes"] == 8


def test_short_file_returns_nan(tmp_path):
    path = tmp_path / "empty.bin"
    write_bin([[1, 2, 3]], path)
    result = evaluate_bpb(UniformModel(8), path, FakeTokenizer(), context_len=8)
    assert math.isnan(result["bpb"])
```

- [ ] **Étape 2 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_eval.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.eval'`

- [ ] **Étape 3 : Implémenter `src/tinyllm/eval.py`**

```python
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tinyllm.data import read_bin


def evaluate_bpb(
    model,
    path: str | Path,
    tokenizer,
    context_len: int,
    batch_size: int = 8,
    max_batches: int | None = None,
) -> dict:
    data = read_bin(path)
    if len(data) <= context_len + 1:
        return {"bpb": float("nan"), "tokens": 0, "bytes": 0}
    n_sequences = (len(data) - 1) // context_len

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    batches = 0
    for start in range(0, n_sequences, batch_size):
        indices = range(start, min(start + batch_size, n_sequences))
        rows = np.stack(
            [data[i * context_len : (i + 1) * context_len + 1] for i in indices]
        ).astype(np.int32)
        batch = mx.array(rows)
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        total_nll += float(
            nn.losses.cross_entropy(logits, targets, reduction="sum")
        )
        total_tokens += targets.size
        for row in np.array(targets):
            total_bytes += sum(tokenizer.token_bytes(int(t)) for t in row)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if total_bytes == 0:
        return {"bpb": float("nan"), "tokens": total_tokens, "bytes": 0}
    return {
        "bpb": total_nll / (math.log(2) * total_bytes),
        "tokens": total_tokens,
        "bytes": total_bytes,
    }


def main() -> None:
    from tinyllm.config import load_config
    from tinyllm.sample import load_run_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--val-bin")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--batches", type=int)
    args = parser.parse_args()

    model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)
    result = evaluate_bpb(
        model,
        args.val_bin or cfg.data.val_bin,
        tokenizer,
        args.context_len or cfg.model.context_len,
        max_batches=args.batches,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Étape 4 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_eval.py -q`
Expected: `2 passed`

- [ ] **Étape 5 : Commit**

```bash
git add src/tinyllm/eval.py tests/test_eval.py
git commit -m "feat: add bits-per-byte evaluation"
```

---
### Tâche 16 : Quantification 4-bit et chargement

**Fichiers :**
- Créer : `src/tinyllm/quantize.py`
- Créer : `tests/test_quantize.py`

- [ ] **Étape 1 : Vérifier l'API MLX disponible**

Run: `uv run python -c "import mlx.nn as nn; print(hasattr(nn, 'quantize'))"`
Expected: `True`. Si `False`, mettre à jour MLX (`uv add 'mlx>=0.22'`) avant de continuer.

- [ ] **Étape 2 : Écrire les tests qui échouent**

```python
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.quantize import load_model, quantize_linears, save_quantized
from tinyllm.tokenizer import train_tokenizer


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=320, d_model=32, n_layers=1, n_heads=4,
        n_kv_heads=2, ffn_hidden=64, context_len=16,
    )


def _model() -> TinyLM:
    mx.random.seed(0)
    return TinyLM(_cfg())


def test_quantize_preserves_shape_and_finiteness():
    model = _model()
    tokens = mx.array([[1, 2, 3, 4]])
    quantized = quantize_linears(model, group_size=32, bits=4)
    out = quantized(tokens)
    assert out.shape == (1, 4, 320)
    assert bool(mx.all(mx.isfinite(out)))


def test_quantized_model_is_smaller():
    model = _model()
    fp32_bytes = sum(p.nbytes for _, p in tree_flatten(model.parameters()))
    quantized = quantize_linears(model, group_size=32, bits=4)
    q_bytes = sum(p.nbytes for _, p in tree_flatten(quantized.parameters()))
    assert q_bytes < fp32_bytes * 0.6


def test_save_load_roundtrip(tmp_path):
    model = _model()
    tokens = mx.array([[1, 2, 3, 4]])
    quantized = quantize_linears(model)
    expected = np.array(quantized(tokens))

    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(["hello world " * 50], vocab_size=320, out_path=tokenizer_path)
    out_dir = tmp_path / "model"
    save_quantized(quantized, _cfg(), tokenizer_path, out_dir)
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "config.json").exists()

    loaded, tokenizer = load_model(out_dir)
    np.testing.assert_allclose(np.array(loaded(tokens)), expected, atol=1e-5)
    assert tokenizer.vocab_size >= 320
```

- [ ] **Étape 3 : Lancer les tests, vérifier l'échec**

Run: `uv run pytest tests/test_quantize.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'tinyllm.quantize'`

- [ ] **Étape 4 : Implémenter le cœur de `src/tinyllm/quantize.py`**

```python
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from tinyllm.config import ModelConfig
from tinyllm.model import TinyLM
from tinyllm.tokenizer import TokenizerWrapper


def quantize_linears(
    model: nn.Module, group_size: int = 64, bits: int = 4
) -> nn.Module:
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, module: isinstance(module, nn.Linear),
    )
    return model


def save_quantized(
    model: TinyLM,
    cfg: ModelConfig,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    bits: int = 4,
    group_size: int = 64,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(weights)
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)
    metadata = {
        "model": asdict(cfg),
        "quantization": {"bits": bits, "group_size": group_size},
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy(tokenizer_path, out_dir / "tokenizer.json")
    size = (out_dir / "model.safetensors").stat().st_size
    return {"size_bytes": size, "size_mb": size / 2**20}


def load_model(
    model_dir: str | Path,
) -> tuple[TinyLM, TokenizerWrapper]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "config.json").read_text())
    cfg = ModelConfig(**metadata["model"])
    quantization = metadata["quantization"]
    model = TinyLM(cfg)
    model = quantize_linears(
        model, group_size=quantization["group_size"], bits=quantization["bits"]
    )
    weights = mx.load(str(model_dir / "model.safetensors"))
    model.load_weights(list(weights.items()))
    return model, TokenizerWrapper(model_dir / "tokenizer.json")


def main() -> None:
    import argparse

    from tinyllm.eval import evaluate_bpb
    from tinyllm.sample import load_run_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--val-bin")
    parser.add_argument("--max-degradation", type=float, default=0.15)
    args = parser.parse_args()

    model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)
    before = None
    if args.val_bin:
        before = evaluate_bpb(
            model, args.val_bin, tokenizer, cfg.model.context_len
        )
    quantized = quantize_linears(model, args.group_size, args.bits)
    report = save_quantized(
        quantized,
        cfg.model,
        cfg.data.tokenizer_path,
        args.out,
        bits=args.bits,
        group_size=args.group_size,
    )
    if before is not None:
        after = evaluate_bpb(
            quantized, args.val_bin, tokenizer, cfg.model.context_len
        )
        degradation = after["bpb"] / before["bpb"] - 1.0
        report.update(
            {
                "bpb_before": before["bpb"],
                "bpb_after": after["bpb"],
                "degradation": degradation,
            }
        )
        if degradation > args.max_degradation:
            raise SystemExit(
                f"BPB degradation {degradation:.1%} exceeds "
                f"{args.max_degradation:.1%}"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Étape 5 : Lancer les tests, vérifier le succès**

Run: `uv run pytest tests/test_quantize.py -q`
Expected: `3 passed`

- [ ] **Étape 6 : Commit**

```bash
git add src/tinyllm/quantize.py tests/test_quantize.py
git commit -m "feat: add 4-bit quantization with save and load"
```

---
### Tâche 17 : Configs, scripts et documentation des sources

**Fichiers :**
- Créer : `configs/phase1_tinystories.json`
- Créer : `configs/phase2_mix.json`
- Créer : `configs/phase2_mix_extended.json`
- Créer : `scripts/train_tokenizer.py`
- Créer : `scripts/prepare_data.py`
- Créer : `scripts/run_phase1.sh`
- Créer : `scripts/run_phase2.sh`
- Créer : `docs/data-sources.md`
- Créer : `tests/test_configs.py`
- Modifier : `README.md`

- [ ] **Étape 1 : Écrire le test des configs**

```python
from pathlib import Path

import pytest

from tinyllm.config import load_config

CONFIGS = [
    "configs/phase1_tinystories.json",
    "configs/phase2_mix.json",
    "configs/phase2_mix_extended.json",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model.vocab_size == 16384
    assert cfg.model.context_len == 512
    assert cfg.model.n_heads % cfg.model.n_kv_heads == 0
    assert cfg.train.max_steps > 0
    assert sum(cfg.data.sources.values()) == pytest.approx(1.0)
    assert Path(path).exists()


def test_phase2_defaults_are_larger_than_phase1():
    phase1 = load_config(CONFIGS[0])
    phase2 = load_config(CONFIGS[1])
    assert phase2.model.d_model > phase1.model.d_model
    assert phase2.model.n_layers > phase1.model.n_layers
```

- [ ] **Étape 2 : Lancer le test, vérifier l'échec**

Run: `uv run pytest tests/test_configs.py -q`
Expected: FAIL avec `FileNotFoundError` sur les configs

- [ ] **Étape 3 : Écrire `configs/phase1_tinystories.json`**

```json
{
  "run_name": "phase1_tinystories",
  "output_dir": "runs",
  "model": {
    "vocab_size": 16384,
    "d_model": 512,
    "n_layers": 12,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden": 1408,
    "context_len": 512
  },
  "train": {
    "micro_batch": 16,
    "grad_accum": 8,
    "lr": 0.0006,
    "max_steps": 7629,
    "eval_interval": 500,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "sample_interval": 1000,
    "max_hours": 48,
    "dtype": "bfloat16"
  },
  "data": {
    "tokenizer_path": "data/tokenizer.json",
    "train_bin": "data/phase1/train.bin",
    "val_bin": "data/phase1/val.bin",
    "test_bin": "data/phase1/test.bin",
    "sources": {"tinystories": 1.0},
    "val_frac": 0.01,
    "test_frac": 0.01
  }
}
```

- [ ] **Étape 4 : Écrire `configs/phase2_mix.json`**

```json
{
  "run_name": "phase2_mix",
  "output_dir": "runs",
  "model": {
    "vocab_size": 16384,
    "d_model": 640,
    "n_layers": 16,
    "n_heads": 10,
    "n_kv_heads": 2,
    "ffn_hidden": 1792,
    "context_len": 512
  },
  "train": {
    "micro_batch": 16,
    "grad_accum": 8,
    "lr": 0.0003,
    "max_steps": 30518,
    "eval_interval": 500,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "sample_interval": 1000,
    "max_hours": 168,
    "dtype": "bfloat16"
  },
  "data": {
    "tokenizer_path": "data/phase2/tokenizer.json",
    "train_bin": "data/phase2/train.bin",
    "val_bin": "data/phase2/val.bin",
    "test_bin": "data/phase2/test.bin",
    "sources": {
      "fineweb_edu": 0.65,
      "wiki_en": 0.1,
      "wiki_fr": 0.1,
      "python": 0.1,
      "tinystories": 0.05
    },
    "val_frac": 0.01,
    "test_frac": 0.01
  }
}
```

- [ ] **Étape 5 : Écrire `configs/phase2_mix_extended.json`**

Même contenu que `phase2_mix.json`, avec `"run_name": "phase2_mix_extended"`,
`"d_model": 768`, `"n_heads": 12`, `"n_kv_heads": 4`,
`"ffn_hidden": 2048`, et `"max_hours": 400`.

- [ ] **Étape 6 : Lancer le test, vérifier le succès**

Run: `uv run pytest tests/test_configs.py -q`
Expected: `5 passed`

- [ ] **Étape 7 : Écrire `scripts/train_tokenizer.py`**

```python
import argparse

from tinyllm.config import load_config
from tinyllm.data import clean_text
from tinyllm.prepare import iter_mixed_documents
from tinyllm.tokenizer import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-chars", type=int, default=2_000_000_000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    budget = args.max_chars
    consumed = 0

    def texts():
        nonlocal consumed
        for _, document in iter_mixed_documents(cfg.data.sources, cfg.train.seed):
            text = clean_text(document["text"])
            consumed += len(text)
            if consumed > budget:
                return
            yield text

    train_tokenizer(
        texts(), vocab_size=cfg.model.vocab_size, out_path=cfg.data.tokenizer_path
    )
    print(f"trained tokenizer on ~{consumed} characters")


if __name__ == "__main__":
    main()
```

- [ ] **Étape 8 : Écrire les wrappers et scripts shell**

`scripts/prepare_data.py` :

```python
from tinyllm.prepare import main

if __name__ == "__main__":
    main()
```

`scripts/run_phase1.sh` :

```bash
#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/train_tokenizer.py --config configs/phase1_tinystories.json
uv run python -m tinyllm.prepare --config configs/phase1_tinystories.json
uv run python -m tinyllm.train --config configs/phase1_tinystories.json
```

`scripts/run_phase2.sh` : même chose avec `configs/phase2_mix.json`.

Run: `chmod +x scripts/run_phase1.sh scripts/run_phase2.sh`

- [ ] **Étape 9 : Écrire `docs/data-sources.md`**

Documenter pour chaque source : nom HuggingFace, licence, volume utilisé,
champ texte, et la commande de repli si le dataset change. Inclure un tableau
avec `tinystories` (CDLA-Sharing-1.0), `fineweb-edu` (ODC-By 1.0),
`wikipedia` (CC BY-SA), `the-stack-smol` (licences par dépôt, usage recherche).
Mentionner que les textes sont filtrés et dédupliqués localement.

- [ ] **Étape 10 : Mettre à jour `README.md`**

Remplacer le contenu par : nom du projet, objectif, prérequis (Apple Silicon,
`uv`), les quatre commandes du pipeline, et un lien vers la spec et le plan.

- [ ] **Étape 11 : Commit**

```bash
git add configs scripts docs/data-sources.md README.md tests/test_configs.py
git commit -m "feat: add run configs, orchestration scripts and data docs"
```

---
### Tâche 18 : Exécution de la phase 1

**Fichiers :**
- Créer : `tests/test_overfit_batch.py`
- Créer : `docs/runs/phase1.md`

- [ ] **Étape 1 : Écrire le test d'overfit d'un batch**

```python
import mlx.core as mx
import mlx.nn as nn
import pytest

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import TinyLM
from tinyllm.train import cast_tree, loss_fn, make_optimizer


@pytest.mark.slow
def test_overfit_single_batch():
    cfg = ModelConfig(
        vocab_size=320, d_model=64, n_layers=2, n_heads=4,
        n_kv_heads=2, ffn_hidden=128, context_len=32,
    )
    mx.random.seed(0)
    model = TinyLM(cfg)
    work = TinyLM(cfg)
    optimizer = make_optimizer(
        TrainConfig(lr=1e-2, max_steps=300, warmup_ratio=0.01)
    )
    loss_and_grad = nn.value_and_grad(work, loss_fn)
    batch = mx.random.randint(0, cfg.vocab_size, (4, 33))
    initial = float(loss_fn(model, batch))
    for _ in range(300):
        work.update(cast_tree(model.parameters(), mx.float32))
        _, grads = loss_and_grad(work, batch)
        optimizer.update(model, grads)
    final = float(loss_fn(model, batch))
    print(f"overfit loss {initial:.3f} -> {final:.4f}")
    assert final < 0.1
```

- [ ] **Étape 2 : Lancer toute la suite de tests**

Run: `uv run pytest -q -m "not slow" && uv run pytest -q -m slow`
Expected: tous les tests passent, overfit sous 0,1

- [ ] **Étape 3 : Entraîner le tokenizer**

Run:

```bash
uv run python scripts/train_tokenizer.py \
  --config configs/phase1_tinystories.json --max-chars 500000000
```

Expected: `trained tokenizer on ~500000000 characters`, et
`data/tokenizer.json` existe. Vérifier :

```bash
uv run python -c "
from tinyllm.tokenizer import TokenizerWrapper
t = TokenizerWrapper('data/tokenizer.json')
print(t.vocab_size, t.encode('Once upon a time'), t.decode(t.encode('Ça va ?')))
"
```

Expected: vocabulaire proche de 16 384, roundtrip correct.

- [ ] **Étape 4 : Préparer les données**

Run:

```bash
uv run python -m tinyllm.prepare --config configs/phase1_tinystories.json
```

Expected: `meta.json` avec `"tokens": {"train": ~450000000, "val": ~4500000}`.
Si le train descend sous 300 M tokens, relancer avec `--force` après avoir
vérifié les filtres (`min_chars`, `max_chars`).

- [ ] **Étape 5 : Mesurer le débit sur un run court**

Run:

```bash
uv run python -m tinyllm.train \
  --config configs/phase1_tinystories.json --max-steps 50
```

Expected: 50 steps terminés, `runs/phase1_tinystories/log.jsonl` non vide.
Relever `tokens_per_s` sur les 10 derniers steps, puis estimer :

```bash
uv run python -c "
import json
rows=[json.loads(l) for l in open('runs/phase1_tinystories/log.jsonl')]
speed=sum(r['tokens_per_s'] for r in rows[-5:])/5
hours=7629*65536/speed/3600
print(f'{speed:.0f} tok/s -> {hours:.1f} h pour 7629 steps')
"
```

Si l'estimation dépasse 48 h, réduire `max_steps` à 4 000 et consigner la
décision dans la spec avant le run complet.

- [ ] **Étape 6 : Lancer le run complet**

Le run doit survivre à la fermeture du terminal et à la veille :

```bash
mkdir -p runs/phase1_tinystories
nohup caffeinate -i uv run python -m tinyllm.train \
  --config configs/phase1_tinystories.json \
  > runs/phase1_tinystories/train.out 2>&1 &
```

Expected: le processus tourne, `runs/phase1_tinystories/log.jsonl` grossit,
un checkpoint apparaît toutes les 500 steps.

- [ ] **Étape 7 : Surveiller sans bloquer**

```bash
tail -f runs/phase1_tinystories/log.jsonl
ls runs/phase1_tinystories/checkpoints/
cat runs/phase1_tinystories/samples/step_001000.txt
```

Critères de santé : loss en baisse régulière, `grad_norm` stable (pas de pics
répétés), `peak_memory_mb` < 12 000, échantillons qui passent du bruit à des
phrases.

- [ ] **Étape 8 : Évaluer le modèle final**

```bash
uv run python -m tinyllm.eval \
  --run-dir runs/phase1_tinystories --checkpoint best \
  > runs/phase1_tinystories/eval_global.json
```

Expected: `bpb` < 1,2 (seuil indicatif). Puis générer trois échantillons :

```bash
uv run python -m tinyllm.sample \
  --run-dir runs/phase1_tinystories --checkpoint best \
  --prompt "Once upon a time" --max-tokens 150 --temperature 0.7
```

- [ ] **Étape 9 : Écrire `docs/runs/phase1.md`**

Contenu : config exacte, durée réelle, tokens/s, courbe de loss (valeurs
toutes les 500 steps), BPB, trois échantillons, et une phrase de verdict.

- [ ] **Étape 10 : Commit**

```bash
git add tests/test_overfit_batch.py docs/runs/phase1.md
git commit -m "docs: record phase 1 run and add overfit sanity test"
```

**Porte de décision :** si BPB > 1,5 ou si les échantillons restent du bruit,
ne pas lancer la phase 2. Vérifier dans l'ordre : roundtrip tokenizer, overfit
d'un batch, fuite train/val, learning rate, puis température d'échantillonnage.

---
### Tâche 19 : Exécution de la phase 2

**Fichiers :**
- Créer : `scripts/make_eval_sets.py`
- Créer : `docs/runs/phase2.md`

- [ ] **Étape 1 : Entraîner le tokenizer sur le mélange final**

Run:

```bash
uv run python scripts/train_tokenizer.py \
  --config configs/phase2_mix.json --max-chars 2000000000
```

Expected: `data/phase2/tokenizer.json` créé. Vérifier le rendement :

```bash
uv run python -c "
from tinyllm.tokenizer import TokenizerWrapper
t = TokenizerWrapper('data/phase2/tokenizer.json')
for text in ['The quick brown fox jumps over the lazy dog.',
             'Le renard brun rapide saute par-dessus le chien paresseux.',
             'def add(a, b):\n    return a + b\n']:
    ids = t.encode(text)
    print(f'{len(text)/len(ids):.2f} chars/token', len(ids))
"
```

Expected: environ 3,5 à 4,5 chars/token sur l'anglais et le français, et un
ratio plus élevé sur le code. Si le français descend sous 2,5, le tokenizer
manque de données FR : relancer avec `--max-chars 1000000000`.

- [ ] **Étape 2 : Préparer le corpus**

Run (peut durer plusieurs heures, utiliser `nohup` si nécessaire) :

```bash
mkdir -p runs
nohup caffeinate -i uv run python -m tinyllm.prepare \
  --config configs/phase2_mix.json > runs/prepare_phase2.out 2>&1 &
```

Expected: `data/phase2/meta.json` avec environ 4 milliards de tokens de train.
Surveiller `ls -lh data/phase2/*.bin`.

- [ ] **Étape 3 : Vérifier le mélange réel**

```bash
uv run python -c "
import json
meta=json.load(open('data/phase2/meta.json'))
total=sum(meta['tokens_by_source'].values())
for name,n in sorted(meta['tokens_by_source'].items()):
    print(f'{name:14s} {100*n/total:5.1f}%  {n/1e9:.2f}B')
"
```

Expected: les parts observées sont à moins de 5 points des cibles de la config.
Si `tinystories` ou `python` dévie fortement, ajuster les poids dans
`configs/phase2_mix.json`, relancer avec `--force`, et consigner l'ajustement.

- [ ] **Étape 4 : Créer les jeux d'évaluation par domaine**

`scripts/make_eval_sets.py` :

```python
import argparse
from pathlib import Path

from tinyllm.config import load_config
from tinyllm.data import PackedWriter, clean_text
from tinyllm.prepare import iter_hf_documents
from tinyllm.tokenizer import TokenizerWrapper

DOMAINS = {
    "en": ["fineweb_edu", "wiki_en"],
    "fr": ["wiki_fr"],
    "code": ["python"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--docs-per-source", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer = TokenizerWrapper(cfg.data.tokenizer_path)
    out_dir = Path(cfg.data.train_bin).parent
    for domain, sources in DOMAINS.items():
        writer = PackedWriter(
            out_dir / f"eval_{domain}.bin", cfg.model.context_len, tokenizer.eos_id
        )
        for source in sources:
            count = 0
            for document in iter_hf_documents(source):
                text = clean_text(document["text"])
                if len(text) < 200:
                    continue
                writer.add_document(tokenizer.encode(text))
                count += 1
                if count >= args.docs_per_source:
                    break
        print(domain, writer.close())


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/make_eval_sets.py --config configs/phase2_mix.json`
Expected: trois fichiers `data/phase2/eval_{en,fr,code}.bin` non vides.

- [ ] **Étape 5 : Mesurer le débit sur 50 steps**

```bash
uv run python -m tinyllm.train \
  --config configs/phase2_mix.json --max-steps 50
```

Puis :

```bash
uv run python -c "
import json
rows=[json.loads(l) for l in open('runs/phase2_mix/log.jsonl')]
speed=sum(r['tokens_per_s'] for r in rows[-5:])/5
hours=30518*65536/speed/3600
print(f'{speed:.0f} tok/s -> {hours:.1f} h')
"
```

Expected: entre 60 et 168 h. Si l'estimation dépasse 168 h, réduire
`max_steps` à 20 000 et consigner la décision.

- [ ] **Étape 6 : Lancer le run complet**

```bash
mkdir -p runs/phase2_mix
nohup caffeinate -i uv run python -m tinyllm.train \
  --config configs/phase2_mix.json \
  > runs/phase2_mix/train.out 2>&1 &
```

Expected: checkpoints toutes les 500 steps, échantillons toutes les 1 000.

- [ ] **Étape 7 : Surveiller**

```bash
tail -f runs/phase2_mix/log.jsonl
ls runs/phase2_mix/checkpoints/
cat runs/phase2_mix/samples/step_005000.txt
```

Vérifier toutes les quelques heures : loss en baisse, `peak_memory_mb` < 13 000,
échantillons qui progressent de mots isolés vers des phrases, puis vers du
français et du code reconnaissables.

- [ ] **Étape 8 : Évaluer le modèle final**

```bash
uv run python -m tinyllm.eval --run-dir runs/phase2_mix --checkpoint best
for domain in en fr code; do
  uv run python -m tinyllm.eval --run-dir runs/phase2_mix --checkpoint best \
    --val-bin data/phase2/eval_$domain.bin
done
```

Expected: BPB global < 1,1 (indicatif). Le BPB du code sera plus élevé que
celui de l'anglais : c'est normal, comparer chaque domaine à lui-même.

- [ ] **Étape 9 : Écrire `docs/runs/phase2.md`**

Contenu : config, durée, tokens/s, courbe de loss, BPB global et par domaine,
six échantillons (3 prompts EN à 0,7 et 1,0 ; 1 prompt FR ; 1 prompt code),
mesures mémoire, et verdict honnête sur les limites.

- [ ] **Étape 10 : Commit**

```bash
git add scripts/make_eval_sets.py docs/runs/phase2.md
git commit -m "docs: record phase 2 run and per-domain evaluation"
```

**Porte de décision :** si le BPB global reste > 1,4 ou si le français et le
code sont du bruit après 10 000 steps, arrêter et analyser la composition du
corpus avant de brûler davantage de calcul.

---
### Tâche 20 : Livraison, quantification finale et rapport

**Fichiers :**
- Modifier : `src/tinyllm/sample.py`
- Modifier : `src/tinyllm/eval.py`
- Créer : `scripts/build_report.py`
- Créer : `docs/report.md`
- Modifier : `README.md`

- [ ] **Étape 1 : Ajouter `--model` à `sample.py`**

Dans `main()`, remplacer `parser.add_argument("--run-dir", required=True)` par :

```python
    parser.add_argument("--run-dir")
    parser.add_argument("--model")
```

Puis remplacer le chargement par :

```python
    if args.model:
        from tinyllm.quantize import load_model

        model, tokenizer = load_model(args.model)
    else:
        if not args.run_dir:
            parser.error("--run-dir or --model is required")
        model, tokenizer, _ = load_run_model(args.run_dir, args.checkpoint)
```

- [ ] **Étape 2 : Ajouter `--model` à `eval.py`**

Dans `main()`, remplacer `parser.add_argument("--run-dir", required=True)` par :

```python
    parser.add_argument("--run-dir")
    parser.add_argument("--model")
```

Puis remplacer le bloc de chargement par :

```python
    if args.model:
        from tinyllm.quantize import load_model

        model, tokenizer = load_model(args.model)
        metadata = json.loads((Path(args.model) / "config.json").read_text())
        context_len = args.context_len or metadata["model"]["context_len"]
        val_bin = args.val_bin or "data/phase2/val.bin"
    else:
        if not args.run_dir:
            parser.error("--run-dir or --model is required")
        model, tokenizer, cfg = load_run_model(args.run_dir, args.checkpoint)
        context_len = args.context_len or cfg.model.context_len
        val_bin = args.val_bin or cfg.data.val_bin

    result = evaluate_bpb(
        model, val_bin, tokenizer, context_len, max_batches=args.batches
    )
```

- [ ] **Étape 3 : Quantifier le meilleur checkpoint**

```bash
mkdir -p models/tinyllm-80m-4bit
uv run python -m tinyllm.quantize \
  --run-dir runs/phase2_mix --checkpoint best \
  --out models/tinyllm-80m-4bit \
  --val-bin data/phase2/val.bin \
  > models/tinyllm-80m-4bit/quantize.json
```

Expected: `size_mb` ≤ 100, `degradation` < 0,15. Si la dégradation dépasse,
essayer `--group-size 32` (poids un peu plus gros, meilleure fidélité) et
recommencer.

- [ ] **Étape 4 : Évaluer le modèle quantifié et sauvegarder les résultats**

```bash
uv run python -m tinyllm.eval --model models/tinyllm-80m-4bit \
  > runs/phase2_mix/eval_global.json
for domain in en fr code; do
  uv run python -m tinyllm.eval --model models/tinyllm-80m-4bit \
    --val-bin data/phase2/eval_$domain.bin \
    > runs/phase2_mix/eval_$domain.json
done
```

- [ ] **Étape 5 : Mesurer la mémoire réelle**

```bash
uv run python -c "
import mlx.core as mx
from tinyllm.quantize import load_model
from tinyllm.sample import generate
mx.metal.reset_peak_memory()
model, tokenizer = load_model('models/tinyllm-80m-4bit')
text = generate(model, tokenizer, 'Once upon a time', max_tokens=100)
mx.eval(model.parameters())
print('peak_metal_mb', mx.metal.get_peak_memory()/2**20)
print('weights_mb', sum(p.nbytes for _, p in __import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(model.parameters()))/2**20)
"
```

Puis mesurer le RSS du process :

```bash
/usr/bin/time -l uv run python -m tinyllm.sample \
  --model models/tinyllm-80m-4bit --prompt "Once upon a time" \
  --max-tokens 100 2>&1 | grep "maximum resident set size"
```

Consigner les deux chiffres dans le rapport, séparément.

- [ ] **Étape 5 bis : Suite d'échantillons qualitative**

`scripts/eval_samples.py` :

```python
import argparse
from pathlib import Path

from tinyllm.quantize import load_model
from tinyllm.sample import generate

PROMPTS = [
    "Once upon a time",
    "The little girl opened the door and",
    "In the morning, the city was",
    "The scientist looked at the data and",
    "Le chat est monté sur la table et",
    "Dans la forêt, il faisait très",
    "def fibonacci(n):",
    "class Stack:\n    def __init__(self):",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, tokenizer = load_model(args.model)
    blocks = []
    for prompt in PROMPTS:
        for temperature in (0.7, 1.0):
            text = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=120,
                temperature=temperature,
                top_p=0.9,
                seed=0,
            )
            blocks.append(f"--- {prompt!r} @ T={temperature}\n{text}\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(blocks))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python scripts/eval_samples.py \
  --model models/tinyllm-80m-4bit --out runs/phase2_mix/eval_samples.txt
```

Puis ajouter à `build_report.py`, après le bloc des échantillons :

```python
    extra = run_dir / "eval_samples.txt"
    if extra.exists():
        lines += ["## Suite qualitative (8 prompts × 2 températures)", "",
                  "```text", extra.read_text(), "```", ""]
```

- [ ] **Étape 6 : Écrire `scripts/build_report.py`**

```python
import argparse
import json
from pathlib import Path


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--phase1-dir")
    parser.add_argument("--out", default="docs/report.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load(run_dir / "config.json", {})
    logs = [
        json.loads(line)
        for line in (run_dir / "log.jsonl").read_text().splitlines()
    ]
    evals = {
        name: load(run_dir / f"eval_{name}.json")
        for name in ("global", "en", "fr", "code")
    }
    quant = load(Path(args.model_dir) / "quantize.json", {})
    phase1 = (
        load(Path(args.phase1_dir) / "eval_global.json")
        if args.phase1_dir
        else None
    )
    samples = sorted((run_dir / "samples").glob("*.txt")) if (run_dir / "samples").exists() else []

    lines = [
        f"# Rapport — {cfg.get('run_name', 'run')}",
        "",
        "## Configuration",
        "",
        f"- Modèle : {cfg.get('model')}",
        f"- Entraînement : {cfg.get('train')}",
        "",
        "## Courbe de loss",
        "",
        "| step | tokens | loss | lr | tokens/s | pic MB |",
        "|---|---|---|---|---|---|",
    ]
    stride = max(1, len(logs) // 20)
    for row in logs[::stride]:
        lines.append(
            f"| {row['step']} | {row['tokens']} | {row['loss']:.4f} | "
            f"{row['lr']:.2e} | {row['tokens_per_s']:.0f} | "
            f"{row['peak_memory_mb']:.0f} |"
        )

    lines += ["", "## Bits-per-byte (modèle quantifié)", "",
              "| domaine | BPB |", "|---|---|"]
    for name, result in evals.items():
        value = f"{result['bpb']:.4f}" if result else "non mesuré"
        lines.append(f"| {name} | {value} |")
    if phase1:
        lines.append(f"| phase 1 (TinyStories) | {phase1['bpb']:.4f} |")

    lines += ["", "## Quantification", "", f"- {quant or {}}", ""]
    if samples:
        lines += ["## Derniers échantillons", "", "```text",
                  samples[-1].read_text(), "```", ""]

    lines += [
        "## Limites",
        "",
        "- Modèle de petite taille : connaissances superficielles, pas de "
        "raisonnement fiable, pas de suivi d'instructions.",
        "- Les 100 Mo désignent les poids quantifiés, pas le RSS du process.",
        "- Les échantillons sont sélectionnés, pas un benchmark.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Étape 7 : Générer le rapport**

```bash
uv run python scripts/build_report.py \
  --run-dir runs/phase2_mix --model-dir models/tinyllm-80m-4bit \
  --phase1-dir runs/phase1_tinystories
```

Expected: `docs/report.md` contient la courbe, les BPB et les échantillons.

- [ ] **Étape 8 : Vérifications finales**

```bash
uv run pytest -q -m "not slow"
uv run pytest -q -m slow
uv run ruff format --check .
uv run ruff check .
```

Expected: tests verts, aucun problème de lint.

- [ ] **Étape 9 : Mettre à jour `README.md` avec les résultats**

Ajouter : tableau des BPB, taille des poids, pic mémoire, deux exemples de
sortie, et la commande de génération avec le modèle quantifié.

- [ ] **Étape 10 : Commit et tag**

```bash
git add src/tinyllm/sample.py src/tinyllm/eval.py scripts/build_report.py \
  docs/report.md README.md
git commit -m "docs: add final report and quantized model workflow"
git tag v0.1.0
```

---
<!-- PLAN-END -->







