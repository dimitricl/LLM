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
