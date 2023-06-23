from dataclasses import dataclass

# Note(manif) SE2(x, y, theta)

@dataclass
class HardwareConfig:
    in_path = "/home/henrik/" + "Downloads/b/"
    T_g0_r0_tuple = (-1.3, 0, 0)  # Release data (Radar 0)
    clockwise = False
