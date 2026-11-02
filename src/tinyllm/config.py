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
    sources: dict[str, float] = field(default_factory=dict)
    val_frac: float = 0.01
    test_frac: float = 0.01
    min_chars: int = 200
    max_chars: int = 100_000
    test_bin: str = "data/phase1/test.bin"


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "phase1"
    output_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str | Path) -> RunConfig:
    raw = json.loads(Path(path).read_text())
    data_raw = raw.get("data", {})
    if not data_raw.get("sources"):
        data_raw["sources"] = {"tinystories": 1.0}
    return RunConfig(
        run_name=raw["run_name"],
        output_dir=raw.get("output_dir", "runs"),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        data=DataConfig(**data_raw),
    )


def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2) + "\n")
