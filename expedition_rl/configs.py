from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

@dataclass
class ExpeditionConfig:
    # World
    width: int = 20
    height: int = 20
    max_steps: int = 200
    start_random_radius: int = 3

    # Starting conditions (high variability)
    start_food_min: int = 20
    start_food_max: int = 80
    start_fuel_min: int = 10
    start_fuel_max: int = 50
    start_health_min: float = 0.6     # team health scalar [0,1]
    start_health_max: float = 1.0
    start_morale_min: float = 0.5
    start_morale_max: float = 1.0

    # Stochastic environment
    base_event_rate: float = 0.05      # per step
    weather_transition_matrix: np.ndarray = field(default_factory=lambda: np.array([
        #   C     W     S
        [0.80, 0.18, 0.02],  # Clear -> ...
        [0.20, 0.65, 0.15],  # Wind  -> ...
        [0.05, 0.35, 0.60],  # Storm -> ...
    ]))
    weather_names: Tuple[str, ...] = ("Clear", "Wind", "Storm")
    weather_speed_penalty: Tuple[float, ...] = (1.0, 0.8, 0.5)  # movement effectiveness
    weather_event_scale: Tuple[float, ...] = (1.0, 1.5, 2.2)    # multiplies event likelihood

    # Terrain risk
    risk_mean: float = 0.2
    risk_std: float = 0.15
    risk_smooth_passes: int = 2

    # Costs and consumption
    food_per_step: float = 1.0
    fuel_move_cost: float = 0.5
    rest_health_gain: float = 0.03
    morale_decay: float = 0.002

    # Rewards
    progress_reward_scale: float = 1.0
    step_penalty: float = -0.02
    injury_penalty: float = -0.5
    event_penalty_scale: float = -0.3
    arrival_bonus: float = 30.0
    surplus_food_bonus_scale: float = 0.05
    surplus_fuel_bonus_scale: float = 0.05

    # Risk modulation by actions
    shortcut_risk_bonus: float = 0.25
    forage_food_gain: Tuple[int, int] = (0, 4)   # randint inclusive
    repair_fuel_gain: Tuple[int, int] = (0, 2)

    # Rendering
    render_compact: bool = True

DEFAULTS = ExpeditionConfig()
