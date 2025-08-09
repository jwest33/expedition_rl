from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional
from .configs import ExpeditionConfig, DEFAULTS as DEFAULT_CONFIG
from .simulators import WeatherSim, make_terrain_risk, sample_event
from .utils import manhattan, move_delta
from .renderers import ascii_map, dashboard
from .terrain import TerrainGenerator, TerrainType

class ExpeditionEnv(gym.Env):
    """
    A compact, interpretable environment for temporal decision-making under variable starts
    and regime-dependent shocks.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: ExpeditionConfig = DEFAULT_CONFIG, seed: int | None = None, use_complex_terrain: bool = True):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.weather = WeatherSim(self.cfg)
        self.use_complex_terrain = use_complex_terrain
        
        # Terrain generation
        if self.use_complex_terrain:
            self.terrain_gen = TerrainGenerator(self.cfg.width, self.cfg.height, seed)
            self.terrain = None
            self.movement_cost = None
        else:
            self.terrain = None
            self.movement_cost = None
        
        self.risk = None
        self.pos = (0,0)
        self.goal = (0,0)
        self.step_count = 0
        self.path_was_created = False

        # Resources / state
        self.food = 0.0
        self.fuel = 0.0
        self.health = 1.0
        self.morale = 1.0

        # Expanded observation space for terrain features
        # [x/W, y/H, goalx/W, goaly/H, food, fuel, health, morale, 
        #  weather_onehot(3), risk_here, dist_norm, time_left_norm, 
        #  terrain_type, movement_cost_here, on_road]
        high = np.array([1,1,1,1,  200,200,1,1,  1,1,1,  1,1,1,  8,3,1], dtype=np.float32)
        low  = np.array([0,0,0,0,    0,  0,0,0,  0,0,0,  0,0,0,  0,0,0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Actions: 0 stay, 1 up,2 down,3 left,4 right,5 forage,6 shortcut,7 repair
        self.action_space = spaces.Discrete(8)

    def _random_start_goal(self):
        if self.use_complex_terrain and self.terrain is not None:
            # Smart placement: Start on roads/settlements when possible
            valid_starts = []
            valid_goals = []
            all_starts = []  # Fallback options
            all_goals = []   # Fallback options
            
            for y in range(self.cfg.height):
                for x in range(self.cfg.width):
                    terrain_type = self.terrain[y, x]
                    # Good starting positions (left side)
                    if x < self.cfg.start_random_radius:
                        all_starts.append((x, y))
                        if terrain_type in [TerrainType.ROAD, TerrainType.SETTLEMENT, TerrainType.PLAINS]:
                            valid_starts.append((x, y))
                    # Good goal positions (right side)
                    if x >= self.cfg.width - self.cfg.start_random_radius - 1:
                        all_goals.append((x, y))
                        if terrain_type in [TerrainType.SETTLEMENT, TerrainType.ROAD, TerrainType.PLAINS]:
                            valid_goals.append((x, y))
            
            # Try to use valid positions first
            if valid_starts and valid_goals:
                # Try up to 10 times to find a connected pair
                for _ in range(10):
                    start = valid_starts[self.rng.integers(0, len(valid_starts))]
                    goal = valid_goals[self.rng.integers(0, len(valid_goals))]
                    
                    # Check if path exists
                    if self.terrain_gen.find_path(self.terrain, start, goal) is not None:
                        self.pos = start
                        self.goal = goal
                        return
            
            # Fallback: use any positions and ensure connectivity
            if all_starts and all_goals:
                start = all_starts[self.rng.integers(0, len(all_starts))]
                goal = all_goals[self.rng.integers(0, len(all_goals))]
            else:
                # Ultimate fallback
                sx = int(self.rng.integers(0, min(self.cfg.start_random_radius, self.cfg.width)))
                sy = int(self.rng.integers(0, self.cfg.height))
                gx = int(self.rng.integers(max(self.cfg.width-1-self.cfg.start_random_radius, 0), self.cfg.width))
                gy = int(self.rng.integers(0, self.cfg.height))
                start = (sx, sy)
                goal = (gx, gy)
            
            self.pos = start
            self.goal = goal
            
            # Ensure there's always a path by creating one if needed
            original_terrain = self.terrain.copy()
            self.terrain = self.terrain_gen.create_guaranteed_path(self.terrain, self.pos, self.goal)
            self.path_was_created = not np.array_equal(original_terrain, self.terrain)
            
            # Regenerate risk and movement cost maps after potential terrain changes
            for y in range(self.cfg.height):
                for x in range(self.cfg.width):
                    terrain_type = self.terrain[y, x]
                    self.risk[y, x] = self.terrain_gen.terrain_risk[terrain_type] + self.rng.normal(0, 0.05)
                    self.risk[y, x] = np.clip(self.risk[y, x], 0.01, 0.9)
                    self.movement_cost[y, x] = self.terrain_gen.terrain_movement_cost[terrain_type]
        else:
            # Original random placement
            sx = int(self.rng.integers(0, min(self.cfg.start_random_radius, self.cfg.width)))
            sy = int(self.rng.integers(0, self.cfg.height))
            gx = int(self.rng.integers(max(self.cfg.width-1-self.cfg.start_random_radius, 0), self.cfg.width))
            gy = int(self.rng.integers(0, self.cfg.height))
            self.pos = (sx, sy)
            self.goal = (gx, gy)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.use_complex_terrain:
                self.terrain_gen = TerrainGenerator(self.cfg.width, self.cfg.height, seed)
        
        self.step_count = 0
        self._last_dist = None
        
        # Generate terrain if using complex terrain
        if self.use_complex_terrain:
            self.terrain, self.risk, self.movement_cost = self.terrain_gen.generate_terrain_features()
        else:
            self.terrain = None
            self.movement_cost = None
            self.risk = make_terrain_risk(self.cfg, self.rng)
        
        self._random_start_goal()
        self.weather.reset(self.rng)
        # Variable starting conditions
        self.food = float(self.rng.integers(self.cfg.start_food_min, self.cfg.start_food_max+1))
        self.fuel = float(self.rng.integers(self.cfg.start_fuel_min, self.cfg.start_fuel_max+1))
        self.health = float(self.rng.uniform(self.cfg.start_health_min, self.cfg.start_health_max))
        self.morale = float(self.rng.uniform(self.cfg.start_morale_min, self.cfg.start_morale_max))
        obs = self._get_obs()
        info = self._get_info(last_event="Reset")
        return obs, info

    def _get_obs(self):
        W,H = self.cfg.width, self.cfg.height
        wx = np.zeros(3, dtype=np.float32); wx[self.weather.state]=1.0
        risk_here = float(self.risk[self.pos[1], self.pos[0]])
        dist = manhattan(self.pos, self.goal)
        time_left = self.cfg.max_steps - self.step_count
        
        # Get terrain features if using complex terrain
        if self.use_complex_terrain and self.terrain is not None:
            terrain_type = float(self.terrain[self.pos[1], self.pos[0]])
            movement_cost_here = float(self.movement_cost[self.pos[1], self.pos[0]])
            on_road = 1.0 if terrain_type == TerrainType.ROAD else 0.0
        else:
            terrain_type = 0.0
            movement_cost_here = 1.0
            on_road = 0.0
        
        obs = np.array([
            self.pos[0]/W, self.pos[1]/H, self.goal[0]/W, self.goal[1]/H,
            self.food, self.fuel, self.health, self.morale,
            wx[0], wx[1], wx[2],
            risk_here, dist/(W+H), time_left/self.cfg.max_steps,
            terrain_type/8.0, movement_cost_here/3.0, on_road
        ], dtype=np.float32)
        return obs

    def _get_info(self, last_event=""):
        dist = manhattan(self.pos, self.goal)
        risk_here = float(self.risk[self.pos[1], self.pos[0]])
        
        info = {
            "pos": self.pos, "goal": self.goal, "dist": float(dist),
            "food": self.food, "fuel": self.fuel,
            "health": self.health, "morale": self.morale,
            "weather": ["Clear","Wind","Storm"][self.weather.state],
            "risk_here": risk_here, "time_left": self.cfg.max_steps - self.step_count,
            "last_event": last_event
        }
        
        # Add terrain info if using complex terrain
        if self.use_complex_terrain and self.terrain is not None:
            terrain_type = self.terrain[self.pos[1], self.pos[0]]
            info["terrain"] = TerrainType(terrain_type).name
            info["movement_cost"] = float(self.movement_cost[self.pos[1], self.pos[0]])
            info["path_was_created"] = self.path_was_created
        
        return info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        self.step_count += 1
        last_event = "None"

        # Consume baseline resources / decay
        self.food = max(0.0, self.food - self.cfg.food_per_step)
        self.morale = max(0.0, self.morale - self.cfg.morale_decay)

        # Weather evolves
        self.weather.step(self.rng)

        # Apply action
        dx, dy = move_delta(action)
        attempted_move = (self.pos[0]+dx, self.pos[1]+dy)
        # Boundary clamp
        attempted_move = (int(max(0, min(self.cfg.width-1, attempted_move[0]))),
                          int(max(0, min(self.cfg.height-1, attempted_move[1]))))

        # Movement effectiveness depends on weather and terrain
        if self.use_complex_terrain and self.movement_cost is not None:
            # Terrain affects fuel cost
            terrain_cost = self.movement_cost[attempted_move[1], attempted_move[0]]
            actual_fuel_cost = self.cfg.fuel_move_cost * terrain_cost
        else:
            actual_fuel_cost = self.cfg.fuel_move_cost
            
        move_ok = (dx != 0 or dy != 0) and self.fuel >= actual_fuel_cost
        if move_ok:
            speed_scale = self.cfg.weather_speed_penalty[self.weather.state]
            
            # Terrain can also affect movement success
            if self.use_complex_terrain and self.terrain is not None:
                terrain_type = self.terrain[attempted_move[1], attempted_move[0]]
                # Mountains are harder to traverse in bad weather
                if terrain_type == TerrainType.MOUNTAIN and self.weather.state > 0:
                    speed_scale *= 0.7
                # Roads improve movement in all weather
                elif terrain_type == TerrainType.ROAD:
                    speed_scale = min(1.0, speed_scale * 1.3)
                    
            moved = (self.rng.random() < speed_scale)  # sometimes weather/terrain blocks move
            self.fuel = max(0.0, self.fuel - actual_fuel_cost)
            if moved:
                self.pos = attempted_move

        # Special actions
        if action == 5:  # forage (trade time for food)
            gain = int(self.rng.integers(self.cfg.forage_food_gain[0], self.cfg.forage_food_gain[1]+1))
            
            # Terrain affects foraging success
            if self.use_complex_terrain and self.terrain is not None:
                terrain_type = self.terrain[self.pos[1], self.pos[0]]
                if terrain_type == TerrainType.FOREST:
                    gain = int(gain * 1.5)  # Forests have more food
                elif terrain_type == TerrainType.RIVER:
                    gain = int(gain * 1.3)  # Rivers provide fish
                elif terrain_type == TerrainType.SETTLEMENT:
                    gain = int(gain * 2)  # Can trade in settlements
                elif terrain_type == TerrainType.DESERT:
                    gain = int(gain * 0.3)  # Desert has little food
                elif terrain_type == TerrainType.MOUNTAIN:
                    gain = int(gain * 0.5)  # Mountains have limited food
                    
            self.food += gain
            last_event = f"Foraged +{gain} food"
        elif action == 6:  # risky shortcut: small random teleport toward goal, higher risk
            sx = np.sign(self.goal[0]-self.pos[0])
            sy = np.sign(self.goal[1]-self.pos[1])
            jump = (int(self.pos[0] + sx*self.rng.integers(1,3)),
                    int(self.pos[1] + sy*self.rng.integers(1,3)))
            self.pos = (int(max(0, min(self.cfg.width-1, jump[0]))),
                        int(max(0, min(self.cfg.height-1, jump[1]))))
        elif action == 7:  # repair/optimize gear => small fuel recovery
            gain = int(self.rng.integers(self.cfg.repair_fuel_gain[0], self.cfg.repair_fuel_gain[1]+1))
            self.fuel += gain
            last_event = f"Repaired +{gain} fuel"
        elif action == 0 and self.food > 0:  # rest if staying
            self.health = min(1.0, self.health + self.cfg.rest_health_gain)

        # Events (stochastic shocks)
        cell_risk = float(self.risk[self.pos[1], self.pos[0]])
        rate = self.cfg.base_event_rate + (self.cfg.shortcut_risk_bonus if action==6 else 0.0)
        evt, mag = sample_event(self.cfg, rate, self.weather.state, cell_risk, self.rng)
        event_penalty = 0.0
        if evt is not None:
            if evt == 0:  # injury
                dh = 0.05 + 0.25*mag
                self.health = max(0.0, self.health - dh)
                event_penalty += self.cfg.injury_penalty * (dh/0.3)
                last_event = f"Injury (-{dh:.2f} health)"
            elif evt == 1:  # gear break (fuel loss)
                loss = 1 + int(3*mag)
                self.fuel = max(0.0, self.fuel - loss)
                event_penalty += self.cfg.event_penalty_scale * (loss/3.0)
                last_event = f"Gear break (-{loss} fuel)"
            elif evt == 2:  # lost: random local shuffle
                ox, oy = self.pos
                self.pos = (int(max(0, min(self.cfg.width-1, ox + int(self.rng.integers(-1,2))))),
                            int(max(0, min(self.cfg.height-1, oy + int(self.rng.integers(-1,2))))))
                event_penalty += self.cfg.event_penalty_scale * (0.5 + mag)
                last_event = "Got lost (position drift)"
            elif evt == 3:  # lucky find
                food_gain = int(self.rng.integers(0,3))
                fuel_gain = int(self.rng.integers(0,2))
                self.food += food_gain; self.fuel += fuel_gain
                last_event = f"Lucky find (+{food_gain} food, +{fuel_gain} fuel)"

        # Morale links to health/resources
        self.morale = max(0.0, min(1.0, self.morale + 0.05*(self.health-0.5) + 0.01*((self.food+self.fuel)/100.0 - 0.5)))

        # Termination conditions
        terminated = False
        arrived = (self.pos == self.goal)
        if arrived:
            terminated = True
        if self.health <= 0.0 or self.food <= 0.0:
            terminated = True
        truncated = (self.step_count >= self.cfg.max_steps)

        # Progress reward
        prev_dist = manhattan(self.pos, self.goal) + (1 if arrived else 0)
        if self._last_dist is None:
            self._last_dist = prev_dist + 1
        progress = float(self._last_dist - prev_dist)
        self._last_dist = prev_dist

        reward = self.cfg.progress_reward_scale * progress + self.cfg.step_penalty + event_penalty

        if arrived:
            reward += self.cfg.arrival_bonus
            reward += self.cfg.surplus_food_bonus_scale * self.food
            reward += self.cfg.surplus_fuel_bonus_scale * self.fuel

        obs = self._get_obs()
        info = self._get_info(last_event=last_event)
        return obs, reward, terminated, truncated, info

    def render(self):
        print(dashboard(self.step_count, self._get_info()))
        if not self.cfg.render_compact:
            print(ascii_map(self.cfg.width, self.cfg.height, self.pos, self.goal, self.risk))
