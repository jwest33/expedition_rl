import numpy as np
from expedition_rl.env import ExpeditionEnv, DEFAULT_CONFIG

def test_env_smoke():
    env = ExpeditionEnv(DEFAULT_CONFIG, seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    print("Smoke test passed.")

if __name__ == "__main__":
    test_env_smoke()
