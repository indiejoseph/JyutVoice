"""TextMelDataModule mixes several datasets (`dataset_path` a list): sets from different builders have different extra columns and are joined on the common ones.

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_datamodule_mix.py`.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("pycantonese")
try:  # before `datasets`: pyarrow imported first loads the system libstdc++, which breaks llvmlite (librosa) on some machines
    import jyutvoice.data.text_mel_datamodule as dm
except Exception as e:  # noqa: BLE001
    pytest.skip(f"datamodule imports unavailable: {e!r}", allow_module_level=True)
datasets = pytest.importorskip("datasets")


def rows(prefix, n, **extra):
    return {
        "id": [f"{prefix}{i}" for i in range(n)], "text": ["我 哋 去 食 飯 。"] * n, "lang": ["yue"] * n, "phone": [None] * n,
        "spk_emb": [[0.1] * 192] * n, "decoder_h": [[[0.0] * 80] * 10] * n, **{k: [v] * n for k, v in extra.items()},
    }  # fmt: skip


def make_module(paths, ratio=0.25):
    return dm.TextMelDataModule(
        name="t", dataset_path=paths, batch_size=2, num_workers=0, pin_memory=False, dataset_valid_ratio=ratio, seed=1, load_durations=False,
        add_blank=True, n_fft=1920, n_feats=80, sample_rate=24000, hop_length=480, win_length=1920, f_min=0, f_max=8000, max_duration=None,
    )  # fmt: skip


def test_list_of_datasets_is_mixed_on_the_common_columns(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    datasets.Dataset.from_dict(rows("a", 6)).save_to_disk(a)
    datasets.Dataset.from_dict(rows("b", 10, source="mce", duration=3.0)).save_to_disk(b)  # extra columns the first has not
    m = make_module([a, b])
    m.setup()
    ids = list(m.trainset.dataset["id"]) + list(m.validset.dataset["id"])
    assert sorted(ids) == sorted([f"a{i}" for i in range(6)] + [f"b{i}" for i in range(10)])
    assert set(m.trainset.dataset.column_names) == {"id", "text", "lang", "phone", "spk_emb", "decoder_h"}


def test_a_single_path_still_keeps_all_its_columns(tmp_path):
    b = str(tmp_path / "b")
    datasets.Dataset.from_dict(rows("b", 8, source="mce", duration=3.0)).save_to_disk(b)
    m = make_module(b)
    m.setup()
    assert "source" in m.trainset.dataset.column_names and len(m.trainset.dataset) + len(m.validset.dataset) == 8


def test_globs_and_directories_can_be_mixed_in_one_list(tmp_path):
    for k in range(2):
        datasets.Dataset.from_dict(rows(f"s{k}_", 4)).save_to_disk(str(tmp_path / "shards" / f"shard_{k}"))
    c = str(tmp_path / "c")
    datasets.Dataset.from_dict(rows("c", 4)).save_to_disk(c)
    m = make_module([str(tmp_path / "shards" / "shard_*"), c])
    m.setup()
    assert len(m.trainset.dataset) + len(m.validset.dataset) == 12
