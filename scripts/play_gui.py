#!/usr/bin/env python
"""
Launch the Expedition RL GUI - an interactive simulation game.
"""

from expedition_rl.gui import ExpeditionGUI
from expedition_rl.configs import ExpeditionConfig

if __name__ == "__main__":
    # You can customize the config here if desired
    config = ExpeditionConfig()
    
    # Launch the GUI
    gui = ExpeditionGUI(config)
    gui.run()