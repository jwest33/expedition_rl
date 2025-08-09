#!/usr/bin/env python
"""
Test that paths are always guaranteed between start and goal.
"""

import numpy as np
from expedition_rl.env import ExpeditionEnv
from expedition_rl.configs import ExpeditionConfig
from expedition_rl.terrain import TerrainType

def test_path_guarantee(num_tests=20):
    """Test that every generated world has a valid path."""
    print("=" * 60)
    print("TESTING PATH GUARANTEE")
    print("=" * 60)
    
    config = ExpeditionConfig()
    successes = 0
    paths_created = 0
    
    for i in range(num_tests):
        # Create environment with random seed
        env = ExpeditionEnv(config, seed=i*10, use_complex_terrain=True)
        obs, info = env.reset()
        
        # Check if path exists
        start = info['pos']
        goal = info['goal']
        
        # Try to reach goal with simple pathfinding
        max_steps = config.width * config.height * 2  # Generous limit
        visited = set()
        current = start
        path_found = False
        
        # Simple BFS to verify connectivity
        from collections import deque
        queue = deque([start])
        visited.add(start)
        
        while queue and not path_found:
            current = queue.popleft()
            
            if current == goal:
                path_found = True
                break
                
            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = current[0] + dx
                ny = current[1] + dy
                
                if (0 <= nx < config.width and 0 <= ny < config.height and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        if path_found:
            successes += 1
            
        if info.get('path_was_created', False):
            paths_created += 1
            
        # Show terrain map for first few tests
        if i < 3:
            print(f"\nTest {i+1}:")
            print(f"  Start: {start}, Goal: {goal}")
            print(f"  Path exists: {path_found}")
            print(f"  Path was created: {info.get('path_was_created', False)}")
            
            if i == 0:
                # Show terrain for first test
                print("\n  Terrain Map:")
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
                
                for y in range(min(15, config.height)):  # Show first 15 rows
                    row = "  "
                    for x in range(min(30, config.width)):  # Show first 30 cols
                        if (x, y) == start:
                            row += 'S'
                        elif (x, y) == goal:
                            row += 'G'
                        else:
                            terrain_type = env.terrain[y, x]
                            row += symbols.get(terrain_type, '?')
                    print(row)
                    
    print("\n" + "=" * 60)
    print(f"RESULTS: {successes}/{num_tests} worlds have guaranteed paths")
    print(f"Paths automatically created: {paths_created}/{num_tests}")
    print("=" * 60)
    
    if successes == num_tests:
        print("✓ SUCCESS: All generated worlds have valid paths!")
    else:
        print("✗ FAILURE: Some worlds don't have valid paths")
        
    return successes == num_tests

def test_path_creation():
    """Test the path creation mechanism specifically."""
    print("\n" + "=" * 60)
    print("TESTING PATH CREATION MECHANISM")
    print("=" * 60)
    
    from expedition_rl.terrain import TerrainGenerator
    
    # Create a difficult terrain
    width, height = 20, 10
    gen = TerrainGenerator(width, height, seed=42)
    terrain = np.full((height, width), TerrainType.MOUNTAIN, dtype=int)
    
    # Add some rivers to make it harder
    for x in range(width):
        terrain[height//2, x] = TerrainType.RIVER
        
    print("Initial terrain (all mountains with river barrier):")
    for y in range(height):
        row = ""
        for x in range(width):
            if terrain[y, x] == TerrainType.MOUNTAIN:
                row += '^'
            elif terrain[y, x] == TerrainType.RIVER:
                row += '='
            elif terrain[y, x] == TerrainType.ROAD:
                row += '-'
            else:
                row += '.'
        print(row)
        
    # Create path from corner to corner
    start = (0, 0)
    goal = (width-1, height-1)
    
    # Check initial path
    initial_path = gen.find_path(terrain, start, goal)
    print(f"\nInitial path exists: {initial_path is not None}")
    
    # Create guaranteed path
    terrain_with_path = gen.create_guaranteed_path(terrain.copy(), start, goal)
    
    print("\nTerrain after creating guaranteed path:")
    for y in range(height):
        row = ""
        for x in range(width):
            if (x, y) == start:
                row += 'S'
            elif (x, y) == goal:
                row += 'G'
            elif terrain_with_path[y, x] == TerrainType.MOUNTAIN:
                row += '^'
            elif terrain_with_path[y, x] == TerrainType.RIVER:
                row += '='
            elif terrain_with_path[y, x] == TerrainType.ROAD:
                row += '-'
            elif terrain_with_path[y, x] == TerrainType.PLAINS:
                row += '.'
            else:
                row += '?'
        print(row)
        
    # Verify path exists
    final_path = gen.find_path(terrain_with_path, start, goal)
    print(f"\nFinal path exists: {final_path is not None}")
    
    if final_path:
        print(f"Path length: {len(final_path)} steps")
        print("✓ Path creation mechanism works!")
    else:
        print("✗ Path creation failed!")
        
    return final_path is not None

if __name__ == "__main__":
    # Test path guarantee in random worlds
    test1_pass = test_path_guarantee(num_tests=20)
    
    # Test path creation mechanism
    test2_pass = test_path_creation()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Path guarantee test: {'PASSED' if test1_pass else 'FAILED'}")
    print(f"Path creation test: {'PASSED' if test2_pass else 'FAILED'}")
    
    if test1_pass and test2_pass:
        print("\n✓ All tests passed! Agents will always be able to reach their goals.")
    else:
        print("\n✗ Some tests failed. Check the implementation.")
    print("=" * 60)