import pygame
import numpy as np
from enum import Enum
from typing import Optional, Tuple
import math
import os
from expedition_rl.env import ExpeditionEnv
from expedition_rl.configs import ExpeditionConfig
from expedition_rl.terrain import TerrainType
from expedition_rl.training_manager import TrainingManager, TrainingConfig

class GameState(Enum):
    MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4
    VICTORY = 5
    AGENT_SELECT = 6
    TRAINING = 7
    TRAINING_CONFIG = 8

class AgentType(Enum):
    HUMAN = 1
    RANDOM = 2
    PPO_TRAINED = 3

class ExpeditionGUI:
    def __init__(self, config: ExpeditionConfig = None):
        pygame.init()
        
        # Store config
        self.cfg = config if config else ExpeditionConfig()
        
        # Display settings
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 800
        self.SIDEBAR_WIDTH = 320
        self.MAP_SIZE = min(self.WINDOW_WIDTH - self.SIDEBAR_WIDTH - 40, self.WINDOW_HEIGHT - 120)
        
        # Create display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Expedition RL - Survival Simulation")
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        self.FPS = 60
        
        # Colors
        self.COLORS = {
            'background': (20, 25, 40),
            'sidebar': (30, 35, 50),
            'panel': (40, 45, 65),
            'text': (220, 220, 230),
            'text_dim': (150, 150, 170),
            'health': (220, 80, 80),
            'food': (120, 180, 80),
            'fuel': (230, 180, 60),
            'morale': (150, 120, 220),
            'player': (100, 150, 250),
            'goal': (250, 200, 50),
            'terrain_low': (60, 120, 60),
            'terrain_mid': (140, 110, 70),
            'terrain_high': (180, 90, 50),
            'grid': (50, 55, 70),
            'clear_sky': (100, 150, 200),
            'wind_sky': (120, 130, 150),
            'storm_sky': (70, 75, 90),
            'button': (60, 65, 85),
            'button_hover': (80, 85, 105),
            'button_pressed': (45, 50, 65),
            'success': (80, 220, 100),
            'danger': (220, 60, 60),
            # Terrain type colors
            'plains': (140, 180, 100),
            'forest': (50, 120, 50),
            'mountain': (140, 120, 110),
            'desert': (220, 200, 140),
            'swamp': (70, 100, 80),
            'river': (80, 140, 200),
            'road': (160, 150, 140),
            'settlement': (180, 160, 120)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 20)
        
        # Game state
        self.state = GameState.MENU
        self.env = ExpeditionEnv(self.cfg)
        self.obs = None
        self.info = None
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.action_history = []
        self.step_count = 0
        
        # Agent settings
        self.agent_type = AgentType.HUMAN
        self.agent_model = None
        self.auto_play = False
        self.auto_play_delay = 200  # milliseconds between AI moves
        self.last_auto_play_time = 0
        
        # Animation
        self.animation_tick = 0
        self.weather_particles = []
        self.event_flash = 0
        self.event_message = ""
        
        # Input handling
        self.keys_pressed = set()
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        
        # Training system
        self.training_manager = TrainingManager()
        self.training_config = TrainingConfig()
        self.training_stats = {}
        self.training_graph_data = []
        self.training_start_time = 0
        
    def load_ppo_model(self, model_name: str = "custom_model"):
        """Load trained PPO model if available"""
        try:
            # Try importing with better error handling
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Set environment variable to avoid TensorBoard issues
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            from stable_baselines3 import PPO
            
            model_path = f"{model_name}.zip"
            if os.path.exists(model_path):
                print("Loading trained PPO model...")
                self.agent_model = PPO.load(model_path, device="cpu")
                print("Model loaded successfully!")
                return True
            else:
                error_msg = f"Model file {model_path} not found. Train a model first using:\npython scripts/train_ppo_simple.py"
                print(error_msg)
                self.event_message = "Model not found - using Random Agent instead"
                self.event_flash = 120
                return False
        except Exception as e:
            error_msg = f"Could not load PPO model: {str(e)}\nUsing Random Agent instead."
            print(error_msg)
            self.event_message = "Model load failed - using Random Agent"
            self.event_flash = 120
            return False
            
    def get_ai_action(self):
        """Get action from AI agent"""
        if self.agent_type == AgentType.RANDOM:
            return self.env.action_space.sample()
        elif self.agent_type == AgentType.PPO_TRAINED and self.agent_model:
            action, _ = self.agent_model.predict(self.obs, deterministic=True)
            return int(action)
        return 0
        
    def reset_game(self):
        self.obs, self.info = self.env.reset()
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.action_history = []
        self.step_count = 0
        self.state = GameState.PLAYING
        self.event_flash = 0
        self.event_message = ""
        self.auto_play = (self.agent_type != AgentType.HUMAN)
        
    def handle_action(self, action: int):
        if self.state != GameState.PLAYING:
            return
            
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.last_reward = reward
        self.step_count += 1
        self.action_history.append(action)
        
        # Handle events
        if self.info['last_event'] != "None" and self.info['last_event'] != "Reset":
            self.event_message = self.info['last_event']
            self.event_flash = 30  # Flash for 30 frames
        
        # Check game end
        if terminated or truncated:
            if self.info['pos'] == self.info['goal']:
                self.state = GameState.VICTORY
            else:
                self.state = GameState.GAME_OVER
                
    def draw_terrain_map(self):
        # Calculate map area
        map_x = 20
        map_y = 60
        cell_size = self.MAP_SIZE // max(self.env.cfg.width, self.env.cfg.height)
        
        # Draw map background
        map_rect = pygame.Rect(map_x - 5, map_y - 5, 
                               cell_size * self.env.cfg.width + 10,
                               cell_size * self.env.cfg.height + 10)
        pygame.draw.rect(self.screen, self.COLORS['panel'], map_rect)
        pygame.draw.rect(self.screen, self.COLORS['grid'], map_rect, 2)
        
        # Draw terrain
        if self.env.use_complex_terrain and self.env.terrain is not None:
            # Draw complex terrain types
            terrain_color_map = {
                TerrainType.PLAINS: self.COLORS['plains'],
                TerrainType.FOREST: self.COLORS['forest'],
                TerrainType.MOUNTAIN: self.COLORS['mountain'],
                TerrainType.DESERT: self.COLORS['desert'],
                TerrainType.SWAMP: self.COLORS['swamp'],
                TerrainType.RIVER: self.COLORS['river'],
                TerrainType.ROAD: self.COLORS['road'],
                TerrainType.SETTLEMENT: self.COLORS['settlement']
            }
            
            for y in range(self.env.cfg.height):
                for x in range(self.env.cfg.width):
                    terrain_type = self.env.terrain[y, x]
                    base_color = terrain_color_map.get(terrain_type, self.COLORS['plains'])
                    
                    # Add risk overlay
                    risk = self.env.risk[y, x]
                    # Darken color based on risk
                    color = tuple(max(0, int(c * (1 - risk * 0.3))) for c in base_color)
                    
                    cell_rect = pygame.Rect(map_x + x * cell_size, 
                                           map_y + y * cell_size,
                                           cell_size, cell_size)
                    pygame.draw.rect(self.screen, color, cell_rect)
                    
                    # Special rendering for certain terrain types
                    if terrain_type == TerrainType.RIVER:
                        # Add flowing water effect
                        wave_offset = (self.animation_tick + x * 10) % 20
                        if wave_offset < 10:
                            pygame.draw.line(self.screen, (100, 160, 220),
                                           (cell_rect.x + 2, cell_rect.centery),
                                           (cell_rect.right - 2, cell_rect.centery), 1)
                    elif terrain_type == TerrainType.ROAD:
                        # Add road markings
                        pygame.draw.line(self.screen, (180, 170, 160),
                                       (cell_rect.centerx, cell_rect.y),
                                       (cell_rect.centerx, cell_rect.bottom), 2)
                    elif terrain_type == TerrainType.MOUNTAIN:
                        # Add peak indicator
                        pygame.draw.lines(self.screen, (160, 140, 130), False,
                                        [(cell_rect.x + cell_size//4, cell_rect.bottom - 2),
                                         (cell_rect.centerx, cell_rect.y + cell_size//4),
                                         (cell_rect.right - cell_size//4, cell_rect.bottom - 2)], 1)
                    elif terrain_type == TerrainType.SETTLEMENT:
                        # Add building indicator
                        building_rect = pygame.Rect(cell_rect.centerx - cell_size//4,
                                                   cell_rect.centery - cell_size//4,
                                                   cell_size//2, cell_size//2)
                        pygame.draw.rect(self.screen, (200, 180, 140), building_rect)
                        pygame.draw.rect(self.screen, (100, 80, 60), building_rect, 1)
                    
                    # Grid lines
                    pygame.draw.rect(self.screen, self.COLORS['grid'], cell_rect, 1)
        else:
            # Original risk-based terrain rendering
            if self.env.risk is not None:
                for y in range(self.env.cfg.height):
                    for x in range(self.env.cfg.width):
                        risk = self.env.risk[y, x]
                        # Interpolate color based on risk
                        if risk < 0.3:
                            color = self.COLORS['terrain_low']
                        elif risk < 0.6:
                            color = self.COLORS['terrain_mid']
                        else:
                            color = self.COLORS['terrain_high']
                        
                        # Add slight variation
                        color = tuple(min(255, c + int(risk * 20)) for c in color)
                        
                        cell_rect = pygame.Rect(map_x + x * cell_size, 
                                               map_y + y * cell_size,
                                               cell_size, cell_size)
                        pygame.draw.rect(self.screen, color, cell_rect)
                        
                        # Grid lines
                        pygame.draw.rect(self.screen, self.COLORS['grid'], cell_rect, 1)
        
        # Draw terrain legend if using complex terrain
        if self.env.use_complex_terrain:
            legend_x = map_x + cell_size * self.env.cfg.width + 20
            legend_y = map_y
            
            legend_title = self.font_tiny.render("Terrain Legend", True, self.COLORS['text'])
            self.screen.blit(legend_title, (legend_x, legend_y))
            legend_y += 22
            
            terrain_legend = [
                (TerrainType.PLAINS, "Plains", self.COLORS['plains']),
                (TerrainType.FOREST, "Forest", self.COLORS['forest']),
                (TerrainType.MOUNTAIN, "Mountain", self.COLORS['mountain']),
                (TerrainType.DESERT, "Desert", self.COLORS['desert']),
                (TerrainType.SWAMP, "Swamp", self.COLORS['swamp']),
                (TerrainType.RIVER, "River", self.COLORS['river']),
                (TerrainType.ROAD, "Road (Fast)", self.COLORS['road']),
                (TerrainType.SETTLEMENT, "Settlement", self.COLORS['settlement']),
            ]
            
            for terrain_type, name, color in terrain_legend:
                # Draw color box
                box_rect = pygame.Rect(legend_x, legend_y, 12, 12)
                pygame.draw.rect(self.screen, color, box_rect)
                pygame.draw.rect(self.screen, self.COLORS['text_dim'], box_rect, 1)
                
                # Draw label
                label = self.font_tiny.render(name, True, self.COLORS['text_dim'])
                self.screen.blit(label, (legend_x + 18, legend_y - 2))
                legend_y += 18
                
                # Stop if running out of space
                if legend_y > map_y + self.MAP_SIZE - 20:
                    break
        
        # Show if emergency path was created
        if self.env.use_complex_terrain and self.info and self.info.get('path_was_created', False):
            path_text = self.font_tiny.render("* Emergency path created *", True, self.COLORS['danger'])
            text_rect = path_text.get_rect(center=(map_x + cell_size * self.env.cfg.width // 2, 
                                                   map_y - 10))
            self.screen.blit(path_text, text_rect)
        
        # Draw goal
        if self.info:
            goal_x, goal_y = self.info['goal']
            goal_rect = pygame.Rect(map_x + goal_x * cell_size + cell_size//4,
                                   map_y + goal_y * cell_size + cell_size//4,
                                   cell_size//2, cell_size//2)
            pygame.draw.rect(self.screen, self.COLORS['goal'], goal_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), goal_rect, 2)
            
            # Pulsing effect for goal
            pulse = abs(math.sin(self.animation_tick * 0.05)) * 0.3 + 0.7
            goal_glow = pygame.Rect(map_x + goal_x * cell_size + 2,
                                   map_y + goal_y * cell_size + 2,
                                   cell_size - 4, cell_size - 4)
            s = pygame.Surface((goal_glow.width, goal_glow.height))
            s.set_alpha(int(50 * pulse))
            s.fill(self.COLORS['goal'])
            self.screen.blit(s, goal_glow)
        
        # Draw player
        if self.info:
            pos_x, pos_y = self.info['pos']
            player_center = (map_x + pos_x * cell_size + cell_size//2,
                           map_y + pos_y * cell_size + cell_size//2)
            pygame.draw.circle(self.screen, self.COLORS['player'], 
                             player_center, cell_size//3)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             player_center, cell_size//3, 2)
            
    def draw_resource_bar(self, x: int, y: int, width: int, height: int,
                          value: float, max_value: float, color: Tuple[int, int, int],
                          label: str):
        # Background
        bar_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (40, 40, 50), bar_rect)
        
        # Fill
        fill_width = int((value / max_value) * width)
        fill_rect = pygame.Rect(x, y, fill_width, height)
        pygame.draw.rect(self.screen, color, fill_rect)
        
        # Border
        pygame.draw.rect(self.screen, self.COLORS['text_dim'], bar_rect, 2)
        
        # Label
        label_text = self.font_tiny.render(label, True, self.COLORS['text'])
        self.screen.blit(label_text, (x, y - 20))
        
        # Value text
        value_text = self.font_tiny.render(f"{value:.1f}/{max_value:.0f}", 
                                          True, self.COLORS['text'])
        text_rect = value_text.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(value_text, text_rect)
        
    def draw_sidebar(self):
        # Sidebar background
        sidebar_rect = pygame.Rect(self.WINDOW_WIDTH - self.SIDEBAR_WIDTH, 0,
                                  self.SIDEBAR_WIDTH, self.WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, self.COLORS['sidebar'], sidebar_rect)
        
        if not self.info:
            return
            
        x_base = self.WINDOW_WIDTH - self.SIDEBAR_WIDTH + 20
        
        # Define fixed sections to prevent overlap
        TITLE_Y = 20
        RESOURCES_Y = 75
        STATS_Y = 340
        AGENT_Y = 540
        CONTROLS_Y = 640
        
        # Title
        title = self.font_medium.render("Expedition Status", True, self.COLORS['text'])
        title_rect = title.get_rect(centerx=self.WINDOW_WIDTH - self.SIDEBAR_WIDTH // 2)
        title_rect.y = TITLE_Y
        self.screen.blit(title, title_rect)
        
        # Draw section divider
        pygame.draw.line(self.screen, self.COLORS['text_dim'], 
                        (x_base, TITLE_Y + 45), (self.WINDOW_WIDTH - 20, TITLE_Y + 45), 1)
        
        # Resources section
        y = RESOURCES_Y
        resources_title = self.font_small.render("Resources", True, self.COLORS['text'])
        self.screen.blit(resources_title, (x_base, y))
        y += 30
        
        self.draw_resource_bar(x_base, y + 20, 260, 22, 
                              self.info['health'], 1.0,
                              self.COLORS['health'], "Health")
        y += 55
        
        self.draw_resource_bar(x_base, y + 20, 260, 22,
                              self.info['food'], 100.0,
                              self.COLORS['food'], "Food")
        y += 55
        
        self.draw_resource_bar(x_base, y + 20, 260, 22,
                              self.info['fuel'], 100.0,
                              self.COLORS['fuel'], "Fuel")
        y += 55
        
        self.draw_resource_bar(x_base, y + 20, 260, 22,
                              self.info['morale'], 1.0,
                              self.COLORS['morale'], "Morale")
        
        # Section divider before stats
        pygame.draw.line(self.screen, self.COLORS['text_dim'], 
                        (x_base, STATS_Y - 10), (self.WINDOW_WIDTH - 20, STATS_Y - 10), 1)
        
        # Stats section at fixed position
        y = STATS_Y
        stats_title = self.font_small.render("Statistics", True, self.COLORS['text'])
        self.screen.blit(stats_title, (x_base, y))
        y += 28
        
        stats = [
            f"Position: {self.info['pos']}",
            f"Goal: {self.info['goal']}",
            f"Distance: {self.info['dist']}",
            f"Time Left: {self.info['time_left']}",
            f"Weather: {self.info['weather']}",
        ]
        
        # Add terrain info if available
        if 'terrain' in self.info:
            stats.append(f"Terrain: {self.info['terrain']}")
            stats.append(f"Move Cost: {self.info.get('movement_cost', 1.0):.1f}x")
        
        stats.extend([
            f"Risk Here: {self.info['risk_here']:.2f}",
            f"Step: {self.step_count}",
            f"Total Reward: {self.total_reward:.2f}",
            f"Last Reward: {self.last_reward:.2f}"
        ])
        
        # Draw stats with bounds checking
        for i, stat in enumerate(stats):
            stat_y = y + (i * 19)
            if stat_y > AGENT_Y - 30:  # Stop before agent section
                break
            stat_text = self.font_tiny.render(stat, True, self.COLORS['text_dim'])
            self.screen.blit(stat_text, (x_base, stat_y))
        
        # Section divider before agent
        pygame.draw.line(self.screen, self.COLORS['text_dim'], 
                        (x_base, AGENT_Y - 10), (self.WINDOW_WIDTH - 20, AGENT_Y - 10), 1)
        
        # Agent info section at fixed position
        y = AGENT_Y
        agent_title = self.font_small.render("Agent Mode", True, self.COLORS['text'])
        self.screen.blit(agent_title, (x_base, y))
        y += 25
        
        agent_text = f"Type: {self.agent_type.name}"
        agent_render = self.font_tiny.render(agent_text, True, self.COLORS['text_dim'])
        self.screen.blit(agent_render, (x_base, y))
        y += 20
        
        if self.auto_play:
            auto_text = "AI Playing (Press H for control)"
            auto_render = self.font_tiny.render(auto_text, True, self.COLORS['success'])
            self.screen.blit(auto_render, (x_base, y))
            y += 20
            
            if self.agent_type != AgentType.HUMAN:
                speed_text = f"Speed: {1000/self.auto_play_delay:.1f} moves/sec"
                speed_render = self.font_tiny.render(speed_text, True, self.COLORS['text_dim'])
                self.screen.blit(speed_render, (x_base, y))
        
        # Section divider before controls
        pygame.draw.line(self.screen, self.COLORS['text_dim'], 
                        (x_base, CONTROLS_Y - 10), (self.WINDOW_WIDTH - 20, CONTROLS_Y - 10), 1)
        
        # Controls help at fixed position
        y = CONTROLS_Y
        controls_title = self.font_small.render("Controls", True, self.COLORS['text'])
        self.screen.blit(controls_title, (x_base, y))
        y += 25
        
        if self.agent_type == AgentType.HUMAN or not self.auto_play:
            controls = [
                "Arrow Keys: Move",
                "Space: Stay/Rest",
                "F: Forage for Food",
                "S: Take Shortcut",
                "R: Repair Gear",
                "P: Pause",
                "ESC: Menu"
            ]
        else:
            controls = [
                "H: Take Manual Control",
                "A: Resume AI Control",
                "+/-: Adjust AI Speed",
                "P: Pause",
                "ESC: Menu"
            ]
        
        # Ensure controls fit in available space
        max_controls = min(len(controls), 7)  # Limit to prevent overflow
        for i in range(max_controls):
            control_text = self.font_tiny.render(controls[i], True, self.COLORS['text_dim'])
            self.screen.blit(control_text, (x_base, y))
            y += 18
            if y > self.WINDOW_HEIGHT - 20:
                break
            
    def draw_weather_effects(self):
        if not self.info:
            return
            
        weather = self.info['weather']
        
        # Weather overlay
        if weather == "Wind":
            # Add wind particles
            if self.animation_tick % 3 == 0:
                self.weather_particles.append({
                    'x': np.random.randint(0, self.WINDOW_WIDTH - self.SIDEBAR_WIDTH),
                    'y': -10,
                    'vx': np.random.uniform(2, 5),
                    'vy': np.random.uniform(1, 3),
                    'life': 120
                })
        elif weather == "Storm":
            # Add storm particles
            if self.animation_tick % 2 == 0:
                self.weather_particles.append({
                    'x': np.random.randint(0, self.WINDOW_WIDTH - self.SIDEBAR_WIDTH),
                    'y': -10,
                    'vx': np.random.uniform(4, 8),
                    'vy': np.random.uniform(3, 6),
                    'life': 100
                })
                
        # Update and draw particles
        new_particles = []
        for particle in self.weather_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            if particle['life'] > 0 and particle['y'] < self.WINDOW_HEIGHT:
                new_particles.append(particle)
                
                # Draw particle
                if weather == "Wind":
                    pygame.draw.line(self.screen, (200, 200, 210),
                                   (particle['x'], particle['y']),
                                   (particle['x'] - particle['vx']*2, 
                                    particle['y'] - particle['vy']*2), 1)
                else:  # Storm
                    pygame.draw.line(self.screen, (150, 150, 180),
                                   (particle['x'], particle['y']),
                                   (particle['x'] - particle['vx']*3, 
                                    particle['y'] - particle['vy']*3), 2)
                    
        self.weather_particles = new_particles
        
        # Weather tint overlay
        if weather != "Clear":
            overlay = pygame.Surface((self.WINDOW_WIDTH - self.SIDEBAR_WIDTH, 
                                    self.WINDOW_HEIGHT))
            if weather == "Wind":
                overlay.fill(self.COLORS['wind_sky'])
                overlay.set_alpha(20)
            else:  # Storm
                overlay.fill(self.COLORS['storm_sky'])
                overlay.set_alpha(40)
            self.screen.blit(overlay, (0, 0))
            
    def draw_event_notification(self):
        if self.event_flash > 0:
            # Create notification box
            notification_width = 400
            notification_height = 60
            x = (self.WINDOW_WIDTH - self.SIDEBAR_WIDTH) // 2 - notification_width // 2
            y = self.WINDOW_HEIGHT - 150
            
            # Background with transparency
            s = pygame.Surface((notification_width, notification_height))
            s.set_alpha(200)
            
            # Color based on event type
            if "injury" in self.event_message.lower() or "break" in self.event_message.lower():
                s.fill(self.COLORS['danger'])
            elif "lucky" in self.event_message.lower():
                s.fill(self.COLORS['success'])
            else:
                s.fill(self.COLORS['panel'])
                
            self.screen.blit(s, (x, y))
            
            # Border
            pygame.draw.rect(self.screen, self.COLORS['text'], 
                           pygame.Rect(x, y, notification_width, notification_height), 3)
            
            # Text
            event_text = self.font_medium.render(self.event_message, True, self.COLORS['text'])
            text_rect = event_text.get_rect(center=(x + notification_width//2, 
                                                   y + notification_height//2))
            self.screen.blit(event_text, text_rect)
            
            self.event_flash -= 1
            
    def draw_menu(self):
        # Darken background
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Menu box
        menu_width = 500
        menu_height = 400
        menu_x = self.WINDOW_WIDTH // 2 - menu_width // 2
        menu_y = self.WINDOW_HEIGHT // 2 - menu_height // 2
        
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], menu_rect)
        pygame.draw.rect(self.screen, self.COLORS['text'], menu_rect, 3)
        
        # Title
        title = self.font_large.render("EXPEDITION RL", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH // 2, menu_y + 60))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_small.render("Survival Simulation", True, self.COLORS['text_dim'])
        subtitle_rect = subtitle.get_rect(center=(self.WINDOW_WIDTH // 2, menu_y + 100))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Buttons
        button_width = 200
        button_height = 50
        button_x = self.WINDOW_WIDTH // 2 - button_width // 2
        
        buttons = [
            ("New Game", menu_y + 160),
            ("Select Agent", menu_y + 220),
            ("Train New Model", menu_y + 280),
            ("Exit", menu_y + 340)
        ]
        
        for text, y in buttons:
            button_rect = pygame.Rect(button_x, y, button_width, button_height)
            
            # Check hover
            hover = button_rect.collidepoint(self.mouse_pos)
            color = self.COLORS['button_hover'] if hover else self.COLORS['button']
            
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, self.COLORS['text'], button_rect, 2)
            
            button_text = self.font_medium.render(text, True, self.COLORS['text'])
            text_rect = button_text.get_rect(center=(button_x + button_width // 2, y + button_height // 2))
            self.screen.blit(button_text, text_rect)
            
            # Handle clicks
            if hover and self.mouse_clicked:
                if text == "New Game":
                    self.reset_game()
                elif text == "Select Agent":
                    self.state = GameState.AGENT_SELECT
                elif text == "Train New Model":
                    self.state = GameState.TRAINING_CONFIG
                elif text == "Exit":
                    return False
                    
        return True
        
    def draw_agent_select(self):
        # Darken background
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Selection box
        menu_width = 600
        menu_height = 500
        menu_x = self.WINDOW_WIDTH // 2 - menu_width // 2
        menu_y = self.WINDOW_HEIGHT // 2 - menu_height // 2
        
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], menu_rect)
        pygame.draw.rect(self.screen, self.COLORS['text'], menu_rect, 3)
        
        # Title
        title = self.font_large.render("SELECT AGENT", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH // 2, menu_y + 60))
        self.screen.blit(title, title_rect)
        
        # Agent options
        agents = [
            (AgentType.HUMAN, "Human Player", "Control the expedition manually", menu_y + 140),
            (AgentType.RANDOM, "Random Agent", "Makes random decisions (for testing)", menu_y + 200),
            (AgentType.PPO_TRAINED, "Trained PPO Agent", "AI trained with reinforcement learning", menu_y + 260)
        ]
        
        # Add custom trained models
        y_offset = 320
        available_models = self.training_manager.get_available_models()
        for model_name in available_models[:3]:  # Show up to 3 custom models
            agents.append((AgentType.PPO_TRAINED, f"Custom: {model_name}", 
                         "Your trained model", menu_y + y_offset))
            y_offset += 60
        
        for agent_type, name, desc, y in agents:
            # Button area
            button_width = 500
            button_height = 60
            button_x = self.WINDOW_WIDTH // 2 - button_width // 2
            button_rect = pygame.Rect(button_x, y, button_width, button_height)
            
            # Check hover and selection
            hover = button_rect.collidepoint(self.mouse_pos)
            selected = (self.agent_type == agent_type)
            
            if selected:
                color = self.COLORS['success']
            elif hover:
                color = self.COLORS['button_hover']
            else:
                color = self.COLORS['button']
                
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, self.COLORS['text'], button_rect, 2)
            
            # Agent name
            name_text = self.font_medium.render(name, True, self.COLORS['text'])
            name_rect = name_text.get_rect(center=(self.WINDOW_WIDTH // 2, y + 20))
            self.screen.blit(name_text, name_rect)
            
            # Description
            desc_text = self.font_tiny.render(desc, True, self.COLORS['text_dim'])
            desc_rect = desc_text.get_rect(center=(self.WINDOW_WIDTH // 2, y + 40))
            self.screen.blit(desc_text, desc_rect)
            
            # Handle clicks
            if hover and self.mouse_clicked:
                self.agent_type = agent_type
                if agent_type == AgentType.PPO_TRAINED:
                    # Check if it's a custom model
                    if "custom" in name:
                        model_name = name.replace("Custom: ", "")
                        if not self.load_ppo_model(model_name):
                            self.agent_type = AgentType.RANDOM  # Fall back to random if model fails
                    else:
                        if not self.load_ppo_model("ppo_expedition"):
                            self.agent_type = AgentType.RANDOM  # Fall back to random if model fails
        
        # Back button
        back_button = pygame.Rect(menu_x + menu_width // 2 - 60, menu_y + 420, 120, 40)
        hover_back = back_button.collidepoint(self.mouse_pos)
        pygame.draw.rect(self.screen, self.COLORS['button_hover'] if hover_back else self.COLORS['button'], back_button)
        pygame.draw.rect(self.screen, self.COLORS['text'], back_button, 2)
        
        back_text = self.font_small.render("Back", True, self.COLORS['text'])
        back_rect = back_text.get_rect(center=back_button.center)
        self.screen.blit(back_text, back_rect)
        
        if hover_back and self.mouse_clicked:
            self.state = GameState.MENU
    
    def draw_training_config(self):
        """Draw training configuration screen."""
        # Darken background
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Config box
        menu_width = 700
        menu_height = 550
        menu_x = self.WINDOW_WIDTH // 2 - menu_width // 2
        menu_y = self.WINDOW_HEIGHT // 2 - menu_height // 2
        
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], menu_rect)
        pygame.draw.rect(self.screen, self.COLORS['text'], menu_rect, 3)
        
        # Title
        title = self.font_large.render("TRAINING CONFIGURATION", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH // 2, menu_y + 50))
        self.screen.blit(title, title_rect)
        
        # Configuration options
        y = menu_y + 120
        x_label = menu_x + 50
        x_value = menu_x + 350
        
        configs = [
            ("Model Name:", self.training_config.model_name),
            ("Training Steps:", f"{self.training_config.total_timesteps:,}"),
            ("Environments:", str(self.training_config.n_envs)),
            ("Learning Rate:", f"{self.training_config.learning_rate:.5f}"),
            ("Batch Size:", str(self.training_config.batch_size)),
            ("Gamma:", f"{self.training_config.gamma:.2f}"),
            ("Clip Range:", f"{self.training_config.clip_range:.2f}"),
        ]
        
        for label, value in configs:
            label_text = self.font_small.render(label, True, self.COLORS['text'])
            self.screen.blit(label_text, (x_label, y))
            
            value_text = self.font_small.render(value, True, self.COLORS['success'])
            self.screen.blit(value_text, (x_value, y))
            y += 40
        
        # Preset buttons
        preset_y = menu_y + 420
        presets = [
            ("Quick (5min)", menu_x + 50, {"total_timesteps": 50000}),
            ("Standard (15min)", menu_x + 200, {"total_timesteps": 200000}),
            ("Deep (1hr)", menu_x + 380, {"total_timesteps": 1000000}),
        ]
        
        for name, x, config in presets:
            button_rect = pygame.Rect(x, preset_y, 140, 40)
            hover = button_rect.collidepoint(self.mouse_pos)
            
            pygame.draw.rect(self.screen, 
                           self.COLORS['button_hover'] if hover else self.COLORS['button'], 
                           button_rect)
            pygame.draw.rect(self.screen, self.COLORS['text'], button_rect, 2)
            
            text = self.font_tiny.render(name, True, self.COLORS['text'])
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
            if hover and self.mouse_clicked:
                for key, val in config.items():
                    setattr(self.training_config, key, val)
        
        # Start/Cancel buttons
        button_y = menu_y + 480
        
        # Start button
        start_button = pygame.Rect(menu_x + 200, button_y, 120, 45)
        start_hover = start_button.collidepoint(self.mouse_pos)
        pygame.draw.rect(self.screen, 
                       self.COLORS['success'] if start_hover else self.COLORS['button'], 
                       start_button)
        pygame.draw.rect(self.screen, self.COLORS['text'], start_button, 2)
        
        start_text = self.font_medium.render("START", True, self.COLORS['text'])
        start_rect = start_text.get_rect(center=start_button.center)
        self.screen.blit(start_text, start_rect)
        
        if start_hover and self.mouse_clicked:
            self.start_training()
        
        # Cancel button
        cancel_button = pygame.Rect(menu_x + 380, button_y, 120, 45)
        cancel_hover = cancel_button.collidepoint(self.mouse_pos)
        pygame.draw.rect(self.screen, 
                       self.COLORS['danger'] if cancel_hover else self.COLORS['button'], 
                       cancel_button)
        pygame.draw.rect(self.screen, self.COLORS['text'], cancel_button, 2)
        
        cancel_text = self.font_medium.render("CANCEL", True, self.COLORS['text'])
        cancel_rect = cancel_text.get_rect(center=cancel_button.center)
        self.screen.blit(cancel_text, cancel_rect)
        
        if cancel_hover and self.mouse_clicked:
            self.state = GameState.MENU
    
    def draw_training_screen(self):
        """Draw training visualization screen."""
        # Background
        self.screen.fill(self.COLORS['background'])
        
        # Title
        title = self.font_large.render("TRAINING IN PROGRESS", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)
        
        # Progress bar
        progress_x = 100
        progress_y = 100
        progress_width = self.WINDOW_WIDTH - 200
        progress_height = 40
        
        # Background bar
        prog_rect = pygame.Rect(progress_x, progress_y, progress_width, progress_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], prog_rect)
        
        # Progress fill
        if self.training_stats:
            progress = self.training_stats.get('progress', 0)
            fill_width = int(progress_width * progress)
            fill_rect = pygame.Rect(progress_x, progress_y, fill_width, progress_height)
            pygame.draw.rect(self.screen, self.COLORS['success'], fill_rect)
            
            # Progress text
            progress_text = f"{int(progress * 100)}% - {self.training_stats.get('timesteps_done', 0):,} / {self.training_stats.get('total_timesteps', 0):,} steps"
            text = self.font_small.render(progress_text, True, self.COLORS['text'])
            text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, progress_y + progress_height // 2))
            self.screen.blit(text, text_rect)
        
        pygame.draw.rect(self.screen, self.COLORS['text'], prog_rect, 3)
        
        # Statistics panel
        stats_x = 100
        stats_y = 180
        
        stats_title = self.font_medium.render("Training Statistics", True, self.COLORS['text'])
        self.screen.blit(stats_title, (stats_x, stats_y))
        
        stats_y += 40
        if self.training_stats:
            # Calculate training speed
            elapsed = (pygame.time.get_ticks() - self.training_start_time) / 1000.0
            steps_per_sec = self.training_stats.get('timesteps_done', 0) / max(elapsed, 1)
            
            stats_lines = [
                f"Parallel Environments: {self.training_config.n_envs}",
                f"Episodes Completed: {self.training_stats.get('episodes_completed', 0)}",
                f"Mean Reward: {self.training_stats.get('mean_reward', 0):.2f}",
                f"Std Reward: {self.training_stats.get('std_reward', 0):.2f}",
                f"Mean Episode Length: {self.training_stats.get('mean_length', 0):.1f}",
                f"Training Speed: {steps_per_sec:.0f} steps/sec",
            ]
            
            for line in stats_lines:
                text = self.font_small.render(line, True, self.COLORS['text_dim'])
                self.screen.blit(text, (stats_x + 20, stats_y))
                stats_y += 30
        
        # Reward graph
        graph_x = 100
        graph_y = 380
        graph_width = self.WINDOW_WIDTH - 200
        graph_height = 250
        
        # Graph background
        graph_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], graph_rect)
        pygame.draw.rect(self.screen, self.COLORS['text_dim'], graph_rect, 2)
        
        # Graph title
        graph_title = self.font_small.render("Episode Rewards (Last 100)", True, self.COLORS['text'])
        title_rect = graph_title.get_rect(center=(graph_x + graph_width // 2, graph_y - 20))
        self.screen.blit(graph_title, title_rect)
        
        # Draw graph data
        if self.training_stats and 'last_rewards' in self.training_stats:
            rewards = self.training_stats['last_rewards']
            if len(rewards) > 1:
                # Normalize rewards to graph height
                min_reward = min(rewards) if rewards else 0
                max_reward = max(rewards) if rewards else 1
                reward_range = max_reward - min_reward if max_reward != min_reward else 1
                
                points = []
                for i, reward in enumerate(rewards):
                    x = graph_x + int((i / (len(rewards) - 1)) * (graph_width - 20)) + 10
                    y = graph_y + graph_height - int(((reward - min_reward) / reward_range) * (graph_height - 20)) - 10
                    points.append((x, y))
                
                # Draw lines
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.COLORS['success'], False, points, 2)
                
                # Draw points
                for point in points:
                    pygame.draw.circle(self.screen, self.COLORS['success'], point, 3)
        
        # Stop button
        stop_button = pygame.Rect(self.WINDOW_WIDTH // 2 - 100, graph_y + graph_height + 40, 200, 50)
        stop_hover = stop_button.collidepoint(self.mouse_pos)
        
        pygame.draw.rect(self.screen, 
                       self.COLORS['danger'] if stop_hover else self.COLORS['button'], 
                       stop_button)
        pygame.draw.rect(self.screen, self.COLORS['text'], stop_button, 2)
        
        stop_text = self.font_medium.render("STOP TRAINING", True, self.COLORS['text'])
        stop_rect = stop_text.get_rect(center=stop_button.center)
        self.screen.blit(stop_text, stop_rect)
        
        if stop_hover and self.mouse_clicked:
            self.stop_training()
    
    def start_training(self):
        """Start the training process."""
        self.state = GameState.TRAINING
        self.training_start_time = pygame.time.get_ticks()
        self.training_graph_data = []
        
        # Start training in background
        success = self.training_manager.start_training(
            env_class=ExpeditionEnv,
            env_config=self.cfg,
            training_config=self.training_config,
            on_update=self.on_training_update,
            on_complete=self.on_training_complete,
            on_error=self.on_training_error
        )
        
        if not success:
            self.state = GameState.MENU
            self.event_message = "Failed to start training"
            self.event_flash = 60
    
    def stop_training(self):
        """Stop the training process."""
        self.training_manager.stop_training()
        self.state = GameState.MENU
        self.event_message = "Training stopped"
        self.event_flash = 60
    
    def on_training_update(self, stats):
        """Handle training progress updates."""
        self.training_stats = stats
        
    def on_training_complete(self, model_path):
        """Handle training completion."""
        self.state = GameState.MENU
        self.event_message = f"Training complete! Model saved as {model_path}"
        self.event_flash = 120
        
        # Reload available models
        self.agent_model = None  # Force reload
        
    def on_training_error(self, error):
        """Handle training errors."""
        self.state = GameState.MENU
        self.event_message = f"Training error: {error}"
        self.event_flash = 120
        
    def draw_game_over(self):
        # Darken background
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Result box
        box_width = 600
        box_height = 300
        box_x = self.WINDOW_WIDTH // 2 - box_width // 2
        box_y = self.WINDOW_HEIGHT // 2 - box_height // 2
        
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        
        if self.state == GameState.VICTORY:
            pygame.draw.rect(self.screen, self.COLORS['success'], box_rect)
            title_text = "MISSION COMPLETE!"
            subtitle_text = f"You reached the goal in {self.step_count} steps!"
        else:
            pygame.draw.rect(self.screen, self.COLORS['danger'], box_rect)
            title_text = "MISSION FAILED"
            if self.info['health'] <= 0:
                subtitle_text = "Your team's health was depleted"
            elif self.info['food'] <= 0:
                subtitle_text = "You ran out of food"
            else:
                subtitle_text = "Time ran out"
                
        pygame.draw.rect(self.screen, self.COLORS['text'], box_rect, 3)
        
        # Text
        title = self.font_large.render(title_text, True, self.COLORS['text'])
        title_rect = title.get_rect(center=(self.WINDOW_WIDTH // 2, box_y + 60))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_medium.render(subtitle_text, True, self.COLORS['text'])
        subtitle_rect = subtitle.get_rect(center=(self.WINDOW_WIDTH // 2, box_y + 120))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Stats
        stats_text = f"Total Reward: {self.total_reward:.2f}"
        stats = self.font_small.render(stats_text, True, self.COLORS['text'])
        stats_rect = stats.get_rect(center=(self.WINDOW_WIDTH // 2, box_y + 180))
        self.screen.blit(stats, stats_rect)
        
        # Continue prompt
        prompt = self.font_small.render("Press SPACE for menu or N for new game", 
                                       True, self.COLORS['text'])
        prompt_rect = prompt.get_rect(center=(self.WINDOW_WIDTH // 2, box_y + 240))
        self.screen.blit(prompt, prompt_rect)
        
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                
                # Handle game controls
                if self.state == GameState.PLAYING:
                    if not self.auto_play:  # Only allow manual controls when not in auto mode
                        if event.key == pygame.K_UP:
                            self.handle_action(1)
                        elif event.key == pygame.K_DOWN:
                            self.handle_action(2)
                        elif event.key == pygame.K_LEFT:
                            self.handle_action(3)
                        elif event.key == pygame.K_RIGHT:
                            self.handle_action(4)
                        elif event.key == pygame.K_SPACE:
                            self.handle_action(0)  # Stay
                        elif event.key == pygame.K_f:
                            self.handle_action(5)  # Forage
                        elif event.key == pygame.K_s:
                            self.handle_action(6)  # Shortcut
                        elif event.key == pygame.K_r:
                            self.handle_action(7)  # Repair
                    
                    # Controls available in both modes
                    if event.key == pygame.K_h:  # Take manual control
                        self.auto_play = False
                    elif event.key == pygame.K_a:  # Resume auto play
                        if self.agent_type != AgentType.HUMAN:
                            self.auto_play = True
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.auto_play_delay = max(50, self.auto_play_delay - 50)
                    elif event.key == pygame.K_MINUS:
                        self.auto_play_delay = min(1000, self.auto_play_delay + 50)
                    elif event.key == pygame.K_p:
                        self.state = GameState.PAUSED
                    elif event.key == pygame.K_ESCAPE:
                        self.state = GameState.MENU
                        
                # Handle menu/game over controls
                elif self.state in [GameState.GAME_OVER, GameState.VICTORY]:
                    if event.key == pygame.K_SPACE:
                        self.state = GameState.MENU
                    elif event.key == pygame.K_n:
                        self.reset_game()
                        
                elif self.state == GameState.PAUSED:
                    if event.key == pygame.K_p or event.key == pygame.K_ESCAPE:
                        self.state = GameState.PLAYING
                        
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
                
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_clicked = True
                
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_clicked = False
                
        return True
        
    def run(self):
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle input
            running = self.handle_input()
            
            # Handle AI auto-play
            if self.state == GameState.PLAYING and self.auto_play:
                if current_time - self.last_auto_play_time > self.auto_play_delay:
                    action = self.get_ai_action()
                    self.handle_action(action)
                    self.last_auto_play_time = current_time
            
            # Clear screen
            self.screen.fill(self.COLORS['background'])
            
            # Draw based on state
            if self.state in [GameState.PLAYING, GameState.PAUSED]:
                self.draw_terrain_map()
                self.draw_sidebar()
                self.draw_weather_effects()
                self.draw_event_notification()
                
                if self.state == GameState.PAUSED:
                    # Pause overlay
                    pause_text = self.font_large.render("PAUSED", True, self.COLORS['text'])
                    text_rect = pause_text.get_rect(center=(self.WINDOW_WIDTH // 2 - self.SIDEBAR_WIDTH // 2,
                                                           self.WINDOW_HEIGHT // 2))
                    self.screen.blit(pause_text, text_rect)
                    
            if self.state == GameState.MENU:
                running = self.draw_menu()
                
            elif self.state == GameState.AGENT_SELECT:
                self.draw_agent_select()
                
            elif self.state == GameState.TRAINING_CONFIG:
                self.draw_training_config()
                
            elif self.state == GameState.TRAINING:
                self.draw_training_screen()
                
            elif self.state in [GameState.GAME_OVER, GameState.VICTORY]:
                self.draw_terrain_map()
                self.draw_sidebar()
                self.draw_game_over()
                
            # Update animation
            self.animation_tick += 1
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.FPS)
            
        pygame.quit()

def main():
    gui = ExpeditionGUI()
    gui.run()

if __name__ == "__main__":
    main()