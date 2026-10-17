import json

import numpy as np

from strike_a_pose.datagen.io import dumps, write_json


def test_dumps_rounding_and_order(tmp_path):
    b = dumps({"b": np.array([1.23456789, -0.0000001]), "a": np.bool_(True), "c": np.int64(3)})
    assert b.endswith(b"\n")
    d = json.loads(b)
    assert list(d) == ["a", "b", "c"] and d["b"] == [1.234568, 0.0] and d["a"] is True
    assert "-0.0" not in b.decode()
    sha = write_json(tmp_path / "x.json", {"k": 1.0})
    assert len(sha) == 64 and not (tmp_path / "x.json.tmp").exists()
