"""Tests for the pure parts of scripts/build_sd3_dataset.py (shard balancing, text validity)."""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("build_sd3", os.path.join(ROOT, "scripts", "build_sd3_dataset.py"))
try:
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
except Exception as e:  # noqa: BLE001  (the script imports the worker module: whisper, onnxruntime, ...)
    bs = None
    _error = repr(e)

pytestmark = pytest.mark.skipif(bs is None, reason="build_sd3_dataset imports unavailable")


def test_every_video_lands_in_exactly_one_shard_and_the_load_is_balanced():
    videos = [(f"v{i}", float(1 + (i * 37) % 50)) for i in range(100)]
    shards = bs.balance_shards(videos, 8)
    flat = [v for s in shards for v in s]
    assert sorted(flat) == sorted(v for v, _ in videos) and len(flat) == len(set(flat))
    seconds = dict(videos)
    load = [sum(seconds[v] for v in s) for s in shards]
    assert max(load) - min(load) <= max(seconds.values())


def test_more_shards_than_videos_leaves_empty_shards_not_errors():
    shards = bs.balance_shards([("a", 5.0), ("b", 3.0)], 4)
    assert sum(len(s) for s in shards) == 2 and len(shards) == 4


@pytest.mark.parametrize(
    "text, ok",
    [("佢哋而家去搵嘢食喎，好唔好呀？", True), ("我 喐 咗 少少 。", True), ("好嘅…「得」", True), ("", False), ("Hello 你好", False),
     ("好,唔好", False), ("我有3個", False), ("你好 OK。", False)],
)  # fmt: skip
def test_valid_text_follows_the_word_seg_rule(text, ok):
    pytest.importorskip("pycantonese")
    assert bs.valid_text(text) is ok
