import pytest

from strike_a_pose.datagen.config import ConfigError, config_from_dict, load_config


def test_defaults_and_files():
    cfg = config_from_dict({"dataset_name": "x", "amass_subsets": ["A"]})
    assert (cfg.num_poses, cfg.image.height, cfg.image.width, cfg.camera_ring.count) == (
        2000,
        256,
        192,
        8,
    )
    assert cfg.output_root == "data/generated"
    for name in ("default", "fixture"):
        assert load_config(f"configs/{name}.yaml").camera_ring.count == 8
    assert load_config("configs/fixture.yaml").num_poses * 8 <= 16


@pytest.mark.parametrize(
    "bad",
    [
        {"dataset_name": "x", "amass_subsets": ["A"], "bogus": 1},
        {"dataset_name": "x y", "amass_subsets": ["A"]},
        {"dataset_name": "x", "amass_subsets": []},
        {"dataset_name": "x", "amass_subsets": ["A"], "num_poses": 0},
        {"dataset_name": "x", "amass_subsets": ["A"], "image": {"height": 10, "depth": 3}},
        {"dataset_name": "x", "amass_subsets": ["A"], "camera_ring": {"focal_px": -1}},
    ],
)
def test_bad_configs(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)
