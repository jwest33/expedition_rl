import numpy as np
from expedition_rl.env import ExpeditionEnv, DEFAULT_CONFIG

if __name__ == "__main__":
    env = ExpeditionEnv(DEFAULT_CONFIG, seed=42)
    obs, info = env.reset()
    total = 0.0
    for _ in range(40):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        env.render()
        total += rew
        if term or trunc:
            break
    print(f"Total reward: {total:.2f}")
