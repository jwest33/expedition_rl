"""
Training manager for integrated RL model training in the GUI.
"""

import os
import time
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

@dataclass
class TrainingConfig:
    """Configuration for training an RL model."""
    model_name: str = "custom_model"
    total_timesteps: int = 100000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
class TrainingCallback(BaseCallback):
    """Callback for tracking training progress."""
    
    def __init__(self, update_callback: Callable, total_timesteps: int):
        super().__init__()
        self.update_callback = update_callback
        self.total_timesteps = total_timesteps
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_rewards = None
        self.current_lengths = None
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update UI every 0.1 seconds
        
    def _on_training_start(self) -> None:
        self.current_rewards = np.zeros(self.training_env.num_envs)
        self.current_lengths = np.zeros(self.training_env.num_envs)
        
    def _on_step(self) -> bool:
        # Track episode statistics
        for i in range(self.training_env.num_envs):
            self.current_rewards[i] += self.locals['rewards'][i]
            self.current_lengths[i] += 1
            
            if self.locals['dones'][i]:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
        
        # Update UI periodically
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.last_update_time = current_time
            
            stats = {
                'timesteps_done': self.num_timesteps,
                'total_timesteps': self.total_timesteps,
                'progress': self.num_timesteps / self.total_timesteps,
                'episodes_completed': len(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
                'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'last_rewards': list(self.episode_rewards)[-10:] if self.episode_rewards else [],
            }
            
            self.update_callback(stats)
            
        return True  # Continue training
        
class TrainingManager:
    """Manages RL model training with real-time updates."""
    
    def __init__(self):
        self.is_training = False
        self.training_thread = None
        self.current_model = None
        self.training_stats = {}
        self.training_history = []
        self.should_stop = False
        self.on_update = None
        self.on_complete = None
        self.on_error = None
        
    def start_training(self, env_class, env_config, training_config: TrainingConfig,
                       on_update: Callable = None, on_complete: Callable = None, 
                       on_error: Callable = None):
        """Start training in a background thread."""
        if not SB3_AVAILABLE:
            if on_error:
                on_error("stable-baselines3 not installed. Run: pip install stable-baselines3")
            return False
            
        if self.is_training:
            if on_error:
                on_error("Training already in progress")
            return False
            
        self.on_update = on_update
        self.on_complete = on_complete
        self.on_error = on_error
        self.should_stop = False
        self.is_training = True
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._train_worker,
            args=(env_class, env_config, training_config)
        )
        self.training_thread.start()
        return True
        
    def _train_worker(self, env_class, env_config, training_config: TrainingConfig):
        """Worker thread for training."""
        try:
            # Create vectorized environment
            def make_env(seed):
                def _init():
                    env = env_class(env_config, seed=seed)
                    return env
                return _init
                
            envs = DummyVecEnv([make_env(i) for i in range(training_config.n_envs)])
            
            # Create PPO model
            self.current_model = PPO(
                "MlpPolicy",
                envs,
                learning_rate=training_config.learning_rate,
                n_steps=training_config.n_steps,
                batch_size=training_config.batch_size,
                n_epochs=training_config.n_epochs,
                gamma=training_config.gamma,
                gae_lambda=training_config.gae_lambda,
                clip_range=training_config.clip_range,
                ent_coef=training_config.ent_coef,
                vf_coef=training_config.vf_coef,
                max_grad_norm=training_config.max_grad_norm,
                verbose=0,
                tensorboard_log=None
            )
            
            # Create callback
            callback = TrainingCallback(
                update_callback=self._on_training_update,
                total_timesteps=training_config.total_timesteps
            )
            
            # Train model
            self.current_model.learn(
                total_timesteps=training_config.total_timesteps,
                callback=callback,
                progress_bar=False
            )
            
            # Save model
            model_path = f"{training_config.model_name}.zip"
            self.current_model.save(model_path)
            
            # Training complete
            self.is_training = False
            if self.on_complete:
                self.on_complete(model_path)
                
        except Exception as e:
            self.is_training = False
            if self.on_error:
                self.on_error(str(e))
                
    def _on_training_update(self, stats: Dict[str, Any]):
        """Handle training updates."""
        self.training_stats = stats
        self.training_history.append({
            'timestep': stats['timesteps_done'],
            'mean_reward': stats['mean_reward'],
            'episodes': stats['episodes_completed']
        })
        
        if self.on_update:
            self.on_update(stats)
            
        # Check if should stop
        if self.should_stop:
            raise KeyboardInterrupt("Training stopped by user")
            
    def stop_training(self):
        """Stop the current training."""
        self.should_stop = True
        if self.training_thread:
            self.training_thread.join(timeout=5)
        self.is_training = False
        
    def get_available_models(self) -> list:
        """Get list of available trained models."""
        models = []
        for file in os.listdir('.'):
            if file.endswith('.zip') and ('expedition' in file.lower() or 'custom' in file.lower()):
                models.append(file[:-4])  # Remove .zip extension
        return models
        
    def delete_model(self, model_name: str) -> bool:
        """Delete a saved model."""
        model_path = f"{model_name}.zip"
        if os.path.exists(model_path):
            os.remove(model_path)
            return True
        return False