"""
Advanced terrain generation with procedural patterns for learning.
"""

import numpy as np
from enum import IntEnum
from typing import Tuple, Optional, List, Set
import math
from collections import deque

class TerrainType(IntEnum):
    PLAINS = 0
    FOREST = 1
    MOUNTAIN = 2
    DESERT = 3
    SWAMP = 4
    RIVER = 5
    ROAD = 6
    SETTLEMENT = 7

class TerrainGenerator:
    """Generates complex, learnable terrain patterns."""
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        
        # Terrain type costs and risks
        self.terrain_movement_cost = {
            TerrainType.PLAINS: 1.0,
            TerrainType.FOREST: 1.5,
            TerrainType.MOUNTAIN: 2.5,
            TerrainType.DESERT: 1.8,
            TerrainType.SWAMP: 2.0,
            TerrainType.RIVER: 3.0,
            TerrainType.ROAD: 0.5,
            TerrainType.SETTLEMENT: 0.8
        }
        
        self.terrain_risk = {
            TerrainType.PLAINS: 0.1,
            TerrainType.FOREST: 0.2,
            TerrainType.MOUNTAIN: 0.4,
            TerrainType.DESERT: 0.3,
            TerrainType.SWAMP: 0.35,
            TerrainType.RIVER: 0.25,
            TerrainType.ROAD: 0.05,
            TerrainType.SETTLEMENT: 0.02
        }
        
        self.terrain_food_bonus = {
            TerrainType.PLAINS: 0.2,
            TerrainType.FOREST: 0.4,
            TerrainType.MOUNTAIN: 0.1,
            TerrainType.DESERT: 0.0,
            TerrainType.SWAMP: 0.3,
            TerrainType.RIVER: 0.5,
            TerrainType.ROAD: 0.0,
            TerrainType.SETTLEMENT: 0.6
        }
        
    def generate_base_terrain(self) -> np.ndarray:
        """Generate base terrain using Perlin-like noise."""
        terrain = np.zeros((self.height, self.width), dtype=int)
        
        # Generate multiple octaves of noise for natural patterns
        elevation = self._generate_perlin_noise(scale=0.1, octaves=3)
        moisture = self._generate_perlin_noise(scale=0.15, octaves=2, offset=1000)
        
        # Classify terrain based on elevation and moisture
        for y in range(self.height):
            for x in range(self.width):
                elev = elevation[y, x]
                moist = moisture[y, x]
                
                if elev > 0.7:  # High elevation
                    terrain[y, x] = TerrainType.MOUNTAIN
                elif elev < 0.3 and moist > 0.6:  # Low and wet
                    terrain[y, x] = TerrainType.SWAMP
                elif moist < 0.3:  # Dry areas
                    terrain[y, x] = TerrainType.DESERT
                elif moist > 0.5 and elev > 0.4:  # Mid elevation, moist
                    terrain[y, x] = TerrainType.FOREST
                else:
                    terrain[y, x] = TerrainType.PLAINS
                    
        return terrain
    
    def _generate_perlin_noise(self, scale: float = 0.1, octaves: int = 1, 
                               offset: float = 0) -> np.ndarray:
        """Generate Perlin-like noise using multiple octaves."""
        noise = np.zeros((self.height, self.width))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amp = 0.5 ** octave
            
            # Generate random gradients
            grid_width = int(self.width * scale * freq) + 2
            grid_height = int(self.height * scale * freq) + 2
            gradients = self.rng.random((grid_height, grid_width, 2)) * 2 - 1
            gradients += offset
            
            # Interpolate to get smooth noise
            for y in range(self.height):
                for x in range(self.width):
                    # Grid coordinates
                    gx = x * scale * freq
                    gy = y * scale * freq
                    
                    # Grid cell corners
                    x0, y0 = int(gx), int(gy)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # Interpolation weights
                    wx = gx - x0
                    wy = gy - y0
                    
                    # Smooth the weights
                    wx = wx * wx * (3 - 2 * wx)
                    wy = wy * wy * (3 - 2 * wy)
                    
                    # Simple bilinear interpolation
                    value = self.rng.random() * amp
                    noise[y, x] += value
                    
        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise
    
    def add_rivers(self, terrain: np.ndarray) -> np.ndarray:
        """Add rivers that flow from mountains to low areas."""
        # Find mountain peaks as river sources
        mountain_mask = (terrain == TerrainType.MOUNTAIN)
        
        # Start 2-3 rivers from mountain areas
        num_rivers = self.rng.integers(2, 4)
        
        for _ in range(num_rivers):
            # Find a mountain cell
            mountain_cells = np.argwhere(mountain_mask)
            if len(mountain_cells) == 0:
                continue
                
            start_idx = self.rng.integers(0, len(mountain_cells))
            y, x = mountain_cells[start_idx]
            
            # Flow downhill with some randomness
            river_length = self.rng.integers(10, 20)
            for _ in range(river_length):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if terrain[y, x] not in [TerrainType.MOUNTAIN, TerrainType.SETTLEMENT]:
                        terrain[y, x] = TerrainType.RIVER
                    
                    # Move generally toward edges with some randomness
                    dx = self.rng.choice([-1, 0, 1], p=[0.3, 0.2, 0.5] if x < self.width/2 else [0.5, 0.2, 0.3])
                    dy = self.rng.choice([-1, 0, 1], p=[0.3, 0.2, 0.5] if y < self.height/2 else [0.5, 0.2, 0.3])
                    
                    x = np.clip(x + dx, 0, self.width - 1)
                    y = np.clip(y + dy, 0, self.height - 1)
                else:
                    break
                    
        return terrain
    
    def add_roads(self, terrain: np.ndarray) -> np.ndarray:
        """Add roads connecting strategic points."""
        # Create roads that generally go from west to east
        # This creates learnable patterns for the agent
        
        # Add 1-2 major roads
        num_roads = self.rng.integers(1, 3)
        
        for road_idx in range(num_roads):
            # Start from western edge
            start_y = self.rng.integers(self.height // 4, 3 * self.height // 4)
            y = start_y
            
            for x in range(self.width):
                # Roads avoid mountains and rivers when possible
                if terrain[y, x] not in [TerrainType.MOUNTAIN, TerrainType.RIVER]:
                    terrain[y, x] = TerrainType.ROAD
                
                # Roads curve slightly but maintain general direction
                if x % 3 == 0:  # Every few steps, potentially change direction
                    dy = self.rng.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    y = np.clip(y + dy, 1, self.height - 2)
                    
        return terrain
    
    def add_settlements(self, terrain: np.ndarray) -> np.ndarray:
        """Add settlements at strategic locations."""
        # Settlements appear near roads and rivers, avoiding mountains
        num_settlements = self.rng.integers(2, 5)
        
        for _ in range(num_settlements):
            # Find suitable locations
            suitable = []
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if terrain[y, x] == TerrainType.PLAINS:
                        # Check for nearby roads or rivers
                        neighbors = terrain[max(0, y-2):min(self.height, y+3),
                                          max(0, x-2):min(self.width, x+3)]
                        has_road = TerrainType.ROAD in neighbors
                        has_river = TerrainType.RIVER in neighbors
                        
                        if has_road or has_river:
                            suitable.append((y, x))
            
            if suitable:
                idx = self.rng.integers(0, len(suitable))
                y, x = suitable[idx]
                
                # Create a small settlement (2x2 or 3x3)
                size = self.rng.choice([2, 3])
                for dy in range(size):
                    for dx in range(size):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if terrain[ny, nx] not in [TerrainType.RIVER, TerrainType.MOUNTAIN]:
                                terrain[ny, nx] = TerrainType.SETTLEMENT
                                
        return terrain
    
    def find_path(self, terrain: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find if a path exists between start and goal using BFS.
        Returns the path if found, None otherwise.
        """
        if start == goal:
            return [start]
            
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            (x, y), path = queue.popleft()
            
            # Check all 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                    
                # Check if already visited
                if (nx, ny) in visited:
                    continue
                    
                # Check if passable (everything except pure impassable terrain)
                # In this case, all terrain types are passable, just with different costs
                # But we could make mountains in storms impassable, etc.
                
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                
                # Check if we reached the goal
                if (nx, ny) == goal:
                    return new_path
                    
                queue.append(((nx, ny), new_path))
                
        return None
    
    def create_guaranteed_path(self, terrain: np.ndarray, start: Tuple[int, int], 
                              goal: Tuple[int, int]) -> np.ndarray:
        """
        Create a road or clear path between start and goal if no path exists.
        """
        # First check if a path already exists
        if self.find_path(terrain, start, goal) is not None:
            return terrain
            
        # Create a simple path - we'll make a road
        x1, y1 = start
        x2, y2 = goal
        
        # Create path using A* style approach but simpler
        current_x, current_y = x1, y1
        
        while (current_x, current_y) != (x2, y2):
            # Move towards goal
            dx = np.sign(x2 - current_x)
            dy = np.sign(y2 - current_y)
            
            # Randomly choose whether to move horizontally or vertically
            if dx != 0 and dy != 0:
                if self.rng.random() < 0.5:
                    current_x += dx
                else:
                    current_y += dy
            elif dx != 0:
                current_x += dx
            else:
                current_y += dy
                
            # Ensure we're in bounds
            current_x = np.clip(current_x, 0, self.width - 1)
            current_y = np.clip(current_y, 0, self.height - 1)
            
            # Place road or clear terrain
            if terrain[current_y, current_x] in [TerrainType.MOUNTAIN, TerrainType.RIVER]:
                # For difficult terrain, place a road
                terrain[current_y, current_x] = TerrainType.ROAD
            elif terrain[current_y, current_x] == TerrainType.SWAMP:
                # Clear swamps to plains
                terrain[current_y, current_x] = TerrainType.PLAINS
                
        return terrain
    
    def generate_terrain_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete terrain with all features."""
        # Generate base terrain
        terrain = self.generate_base_terrain()
        
        # Add features in order
        terrain = self.add_rivers(terrain)
        terrain = self.add_roads(terrain)
        terrain = self.add_settlements(terrain)
        
        # Calculate risk map based on terrain
        risk_map = np.zeros((self.height, self.width), dtype=np.float32)
        movement_cost = np.ones((self.height, self.width), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                terrain_type = terrain[y, x]
                risk_map[y, x] = self.terrain_risk[terrain_type]
                movement_cost[y, x] = self.terrain_movement_cost[terrain_type]
                
                # Add some noise to make it less uniform
                risk_map[y, x] += self.rng.normal(0, 0.05)
                risk_map[y, x] = np.clip(risk_map[y, x], 0.01, 0.9)
                
        # Smooth risk map slightly for more natural transitions
        try:
            from scipy.ndimage import gaussian_filter
            risk_map = gaussian_filter(risk_map, sigma=0.5)
        except ImportError:
            # If scipy not available, do simple averaging
            risk_map_smooth = risk_map.copy()
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    risk_map_smooth[y, x] = np.mean([
                        risk_map[y-1:y+2, x-1:x+2]
                    ])
            risk_map = risk_map_smooth
        
        return terrain, risk_map, movement_cost
    
    def get_terrain_info(self, x: int, y: int, terrain: np.ndarray) -> dict:
        """Get information about a specific terrain cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            terrain_type = terrain[y, x]
            return {
                'type': TerrainType(terrain_type).name,
                'risk': self.terrain_risk[terrain_type],
                'movement_cost': self.terrain_movement_cost[terrain_type],
                'food_bonus': self.terrain_food_bonus[terrain_type]
            }
        return None