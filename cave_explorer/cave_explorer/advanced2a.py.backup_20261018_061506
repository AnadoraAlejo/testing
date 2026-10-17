"""Advanced 2: map-frame accessible cave geometry, no ROS imports.

Clearance is distance to nearest unknown/occupied/out-of-bounds cell (metres).
A bottleneck is a low-clearance medial-ridge candidate with free travel space
on both sides; it is a useful *candidate*, not a topological proof of a choke.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np

# STORE CAVE FEATURES THAT HAVE BEEN IDENTIFIED

@dataclass(frozen=True) 
class Feature:
    x: float
    y: float
    clearance_m: float
    col: int
    row: int

# DEFINE GEOMOETRY ANALYSER

class GeometryAnalyser:
    def __init__(self, robot_radius=0.5, safety_margin=0.25, max_bottleneck_m=3.0):

        # MIN CLEARANCE = ROBOT R + SAFETY MARGIN = 0.5 + 0.25 + 0.75
        self.min_clearance = float(robot_radius) + float(safety_margin)

        # STORE MAX BOTTLENECK CLEARANCE
        self.max_bottleneck = float(max_bottleneck_m)

# GRID COORD -> MAP COORDS

    @staticmethod
    def transform(col, row, resolution, origin):

        # MAP ORIGIN || X Y AND SPING
        ox = origin[0]
        oy = origin[1]
        theta = origin[2]

        # GRID COORD -> PHYSICAL DISATNCE
        u = (col + 0.5) * resolution
        v = (row + 0.5) * resolution

        # APPLY MAP SPIN AND OFFEST THE ORIGIN TO MAITAIN DURING MAP ROTATION
        x_rotation = math.cos(theta) * u
        x_offset = math.sin(theta) * v
        map_x = ox + x_rotation - x_offset

        y_rotation = math.sin(theta) * u
        y_offset = math.cos(theta) * v
        map_y = oy + y_rotation + y_offset

        return map_x, map_y

# LOCATE NEARBY SAFE GRID CELL

    @staticmethod
    def _nearby_component(safe, col, row):

        # CHECK ROBOT COORD INSIDE MAP ++ SAFE
        row_inside_map = 0 <= row < safe.shape[0]
        col_inside_map = 0 <= col < safe.shape[1]

        if row_inside_map and col_inside_map:
            if safe[row, col]:
                return col, row

        # LOCATE COLUMN AND ROW POSITIONS OF SAFE SELSL
        safe_cells = np.where(safe)
        ys = safe_cells[0]
        xs = safe_cells[1]

        # IF NO SAFE SELL THEN RETURN NOTHING
        if len(xs) == 0:
            return None

        # SQAURE DIFF BETWEEN ROBOT AND SAFE CELL
        col_difference = xs - col
        row_difference = ys - row
        col_distance_squared = col_difference ** 2
        row_distance_squared = row_difference ** 2
        total_distance_squared = col_distance_squared + row_distance_squared
        idx = np.argmin(total_distance_squared)

        # RETURN CLOSEST SAFE CELL
    
        closest_col = int(xs[idx])
        closest_row = int(ys[idx])
        return closest_col, closest_row

# CAVE OCCUPANCY GRID

    def analyse(self, data, width, height, resolution, origin, robot_xy):