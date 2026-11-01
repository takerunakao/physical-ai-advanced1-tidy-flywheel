from tidy_flywheel.sim import TidyEnv
from tidy_flywheel.teacher import run_teacher_episode


def test_teacher_succeeds_on_a_few_seeds():
    env = TidyEnv(image_size=(64, 64))
    for seed in (0, 1, 2):
        r = run_teacher_episode(env, seed)
        assert r["success"], f"seed {seed} failed at phase {r['final_phase']}"
        assert not r["robot_contact"]
        assert r["states"].shape[0] == r["actions"].shape[0] == r["env_states"].shape[0] == r["steps"]
