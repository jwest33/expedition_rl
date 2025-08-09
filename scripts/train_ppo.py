# Note: Requires stable-baselines3 and gymnasium.
# pip install stable-baselines3 gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from expedition_rl.env import ExpeditionEnv, DEFAULT_CONFIG

def make_env(seed=0):
    def _f():
        return ExpeditionEnv(DEFAULT_CONFIG, seed=seed)
    return _f

if __name__ == "__main__":
    env = DummyVecEnv([make_env(i) for i in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs")
    model.learn(total_timesteps=200_000)
    model.save("ppo_expedition")
    print("Saved model to ppo_expedition.zip")
