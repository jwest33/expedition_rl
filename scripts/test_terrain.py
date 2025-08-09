#!/usr/bin/env python
"""
Test the complex terrain generation system.
"""

import numpy as np
from expedition_rl.terrain import TerrainGenerator, TerrainType

def print_terrain_stats(terrain, risk_map, movement_cost):
    """Print statistics about the generated terrain."""
    unique, counts = np.unique(terrain, return_counts=True)
    total_cells = terrain.size
    
    print("\nTerrain Distribution:")
    print("-" * 40)
    for terrain_type, count in zip(unique, counts):
        percentage = (count / total_cells) * 100
        name = TerrainType(terrain_type).name
        print(f"{name:12} : {count:4} cells ({percentage:5.1f}%)")
    
    print("\nRisk Statistics:")
    print("-" * 40)
    print(f"Min Risk     : {risk_map.min():.3f}")
    print(f"Max Risk     : {risk_map.max():.3f}")
    print(f"Mean Risk    : {risk_map.mean():.3f}")
    print(f"Std Risk     : {risk_map.std():.3f}")
    
    print("\nMovement Cost Statistics:")
    print("-" * 40)
    print(f"Min Cost     : {movement_cost.min():.2f}x")
    print(f"Max Cost     : {movement_cost.max():.2f}x")
    print(f"Mean Cost    : {movement_cost.mean():.2f}x")
    
def visualize_terrain_ascii(terrain, width, height):
    """Create ASCII visualization of terrain."""
    symbols = {
        TerrainType.PLAINS: '.',
        TerrainType.FOREST: 'T',
        TerrainType.MOUNTAIN: '^',
        TerrainType.DESERT: '~',
        TerrainType.SWAMP: '#',
        TerrainType.RIVER: '=',
        TerrainType.ROAD: '-',
        TerrainType.SETTLEMENT: 'O'
    }
    
    print("\nTerrain Map (ASCII):")
    print("-" * (width + 2))
    for y in range(height):
        row = "|"
        for x in range(width):
            terrain_type = terrain[y, x]
            row += symbols.get(terrain_type, '?')
        row += "|"
        print(row)
    print("-" * (width + 2))
    
    print("\nLegend:")
    for terrain_type, symbol in symbols.items():
        print(f"  {symbol} = {terrain_type.name}")

if __name__ == "__main__":
    # Test parameters
    width, height = 30, 20
    seed = 42
    
    print("=" * 50)
    print("EXPEDITION RL - TERRAIN GENERATION TEST")
    print("=" * 50)
    
    # Generate terrain
    print(f"\nGenerating {width}x{height} terrain with seed {seed}...")
    generator = TerrainGenerator(width, height, seed)
    terrain, risk_map, movement_cost = generator.generate_terrain_features()
    
    # Display results
    print_terrain_stats(terrain, risk_map, movement_cost)
    visualize_terrain_ascii(terrain, width, height)
    
    # Test specific location info
    print("\n\nSample Location Info:")
    print("-" * 40)
    test_positions = [(0, 0), (width//2, height//2), (width-1, height-1)]
    for x, y in test_positions:
        info = generator.get_terrain_info(x, y, terrain)
        if info:
            print(f"Position ({x:2}, {y:2}): {info['type']:12} | "
                  f"Risk: {info['risk']:.2f} | Move Cost: {info['movement_cost']:.1f}x | "
                  f"Food Bonus: {info['food_bonus']:.1f}")
    
    # Generate multiple worlds to show variety
    print("\n\nGenerating 3 different worlds to show procedural variety:")
    print("=" * 50)
    
    for i in range(3):
        print(f"\nWorld {i+1} (seed={i*100}):")
        gen = TerrainGenerator(20, 15, i*100)
        t, _, _ = gen.generate_terrain_features()
        visualize_terrain_ascii(t, 20, 15)
        
    print("\n" + "=" * 50)
    print("Terrain generation test complete!")
    print("\nKey patterns for agent learning:")
    print("- Roads (----) provide fastest travel with lowest risk")
    print("- Settlements (O) offer food and safety")
    print("- Rivers (====) provide food but slow movement")
    print("- Mountains (^^^) are high risk and slow")
    print("- Following roads is generally optimal strategy")
    print("- Shortcuts through terrain trade risk for distance")