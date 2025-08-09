#!/usr/bin/env python
"""
Simplified PPO training script without TensorBoard to avoid compatibility issues.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from expedition_rl.env import ExpeditionEnv, ExpeditionConfig

def make_env(seed=0):
    def _f():
        return ExpeditionEnv(ExpeditionConfig(), seed=seed)
    return _f

if __name__ == "__main__":
    print("Creating environment...")
    # Training environment with 4 parallel instances
    env = DummyVecEnv([make_env(i) for i in range(4)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(100)])
    
    print("Initializing PPO model...")
    # Create PPO model without tensorboard
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None  # Disable tensorboard
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="ppo_expedition"
    )
    
    print("Starting training...")
    print("This will train for 200,000 timesteps (about 5-10 minutes)")
    print("-" * 50)
    
    # Train the model
    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    print("\n" + "-" * 50)
    print("Training complete!")
    
    # Save the final model
    model.save("ppo_expedition")
    print("Model saved to ppo_expedition.zip")
    
    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    total_rewards = [0.0] * 4
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        for i, (r, d) in enumerate(zip(rewards, dones)):
            total_rewards[i] += r
            if d:
                print(f"Environment {i} finished with total reward: {total_rewards[i]:.2f}")
                total_rewards[i] = 0.0
    
    print("\nTraining complete! You can now use the trained model in the GUI.")
    print("Run: python scripts/play_gui.py and select 'Trained PPO Agent'")