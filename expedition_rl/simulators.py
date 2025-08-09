from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from .configs import ExpeditionConfig

RNG = np.random.default_rng

@dataclass
class WeatherSim:
    config: ExpeditionConfig
    state: int = 0  # 0 Clear, 1 Wind, 2 Storm

    def reset(self, rng: np.random.Generator):
        self.state = int(rng.integers(0, 3))
        return self.state

    def step(self, rng: np.random.Generator):
        P = self.config.weather_transition_matrix
        self.state = int(rng.choice(3, p=P[self.state]))
        return self.state

def make_terrain_risk(cfg: ExpeditionConfig, rng: np.random.Generator) -> np.ndarray:
    risk = rng.normal(cfg.risk_mean, cfg.risk_std, size=(cfg.height, cfg.width)).clip(0.0, 1.0)
    # Smooth to create blobs
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float)
    kernel /= kernel.sum()
    for _ in range(cfg.risk_smooth_passes):
        padded = np.pad(risk, ((1,1),(1,1)), mode="edge")
        out = np.zeros_like(risk)
        for y in range(risk.shape[0]):
            for x in range(risk.shape[1]):
                out[y,x] = (padded[y:y+3, x:x+3] * kernel).sum()
        risk = out
    return risk.clip(0.0, 1.0)

def sample_event(cfg: ExpeditionConfig, base_rate: float, weather_state: int, cell_risk: float, rng: np.random.Generator):
    # Event intensity influenced by weather and terrain risk
    rate = base_rate * cfg.weather_event_scale[weather_state] * (1.0 + cell_risk)
    if rng.random() < rate:
        # 0 injury, 1 gear break (fuel loss), 2 lost (teleport small), 3 lucky find (food/fuel)
        evt = int(rng.choice(4, p=[0.35, 0.30, 0.20, 0.15]))
        magnitude = float(rng.random())
        return evt, magnitude
    return None, 0.0
