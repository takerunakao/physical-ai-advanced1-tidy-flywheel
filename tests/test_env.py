import numpy as np

from tidy_flywheel.sim import TidyEnv


def test_reset_is_deterministic():
    env = TidyEnv(image_size=(64, 64))
    _, i1 = env.reset(seed=3)
    _, i2 = env.reset(seed=3)
    assert np.allclose(i1["cube_pos"], i2["cube_pos"]) and i1["cube_yaw"] == i2["cube_yaw"]
    _, i3 = env.reset(seed=4)
    assert not np.allclose(i1["cube_pos"], i3["cube_pos"])


def test_step_tracks_command_and_renders():
    env = TidyEnv(image_size=(64, 64))
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (5,) and obs["images"]["front"].shape == (64, 64, 3)
    for _ in range(10):
        obs, info = env.step(np.array([0.02, 0.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(info["tcp_pos"] - info["tcp_cmd"]) < 0.01
    assert not info["success"]
