from __future__ import annotations

def clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def move_delta(action: int):
    # 0 stay, 1 up, 2 down, 3 left, 4 right, 5 forage, 6 shortcut, 7 repair
    return {
        0: (0,0),
        1: (0,-1),
        2: (0,1),
        3: (-1,0),
        4: (1,0),
        5: (0,0),
        6: (0,0),
        7: (0,0),
    }[action]
