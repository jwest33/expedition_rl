from __future__ import annotations
from typing import Dict
import numpy as np

def ascii_map(world_w, world_h, pos, goal, risk_map, window=7):
    x, y = pos
    gX, gY = goal
    half = window // 2
    xs = max(0, x-half); xe = min(world_w, x+half+1)
    ys = max(0, y-half); ye = min(world_h, y+half+1)
    out = []
    for yy in range(ys, ye):
        row = []
        for xx in range(xs, xe):
            if (xx,yy)==(x,y):
                row.append("A")
            elif (xx,yy)==(gX,gY):
                row.append("G")
            else:
                r = risk_map[yy, xx]
                if r < 0.15: row.append(".")
                elif r < 0.3: row.append(":")
                elif r < 0.5: row.append("*")
                elif r < 0.7: row.append("x")
                else: row.append("#")
        out.append("".join(row))
    return "\n".join(out)

def dashboard(step, info: Dict):
    # Compact textual dashboard
    return (
        f"Step {step} | Pos {info['pos']} -> Goal {info['goal']} | Dist {info['dist']:.1f}\n"
        f"Food {info['food']:.1f} | Fuel {info['fuel']:.1f} | Health {info['health']:.2f} | Morale {info['morale']:.2f}\n"
        f"Weather {info['weather']} | RiskHere {info['risk_here']:.2f} | TimeLeft {info['time_left']}\n"
        f"LastEvent: {info.get('last_event','None')}"
    )
