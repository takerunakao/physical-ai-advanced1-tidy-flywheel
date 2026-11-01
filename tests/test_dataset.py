import numpy as np

from tidy_flywheel.dataset import DatasetWriter
from tidy_flywheel.sim import TidyEnv
from tidy_flywheel.teacher import run_teacher_episode


def test_dataset_round_trip(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env = TidyEnv(image_size=(64, 64))
    w = DatasetWriter("test/tidy", tmp_path / "ds", 64)
    r = run_teacher_episode(env, 0)
    for t in range(r["steps"]):
        w.add_step(r["images"][t], r["states"][t], r["actions"][t], r["env_states"][t])
    w.end_episode({"seed": 0})
    w.close()
    ds = LeRobotDataset("test/tidy", root=tmp_path / "ds")
    assert ds.num_episodes == 1 and ds.num_frames == r["steps"]
    x = ds[5]
    assert tuple(x["observation.images.front"].shape) == (3, 64, 64)
    assert np.allclose(x["action"].numpy(), r["actions"][5], atol=1e-6)
