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
        # VALIDATE MAP DIMENSIONS AND GRID DATA
        if resolution <= 0 or width < 3 or height < 3 or len(data) != width * height:
            raise ValueError('Invalid occupancy grid dimensions/data/resolution')

        # CONVERT OCCUPANCY DATA INTO A TWO DIMENSIONAL GRID
        grid_values = np.asarray(data, dtype=np.int16)
        grid = grid_values.reshape(height, width)

        # IDENTIFY FREE CELLS (0 TO 29), EXCLUDING UNKNOWN AND OCCUPIED CELLS
        known = grid >= 0
        low_occupancy = grid < 30
        free = known & low_occupancy

        # ADD A BORDER SO THE OUTSIDE OF THE MAP COUNTS AS BLOCKED
        padded = np.pad(free.astype(np.uint8), 1)

        # CALCULATE DISTANCE FROM EACH FREE CELL TO NEAREST BLOCKED CELL
        distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        clearance_cells = distances[1:-1, 1:-1]
        clearance = clearance_cells * resolution

        # KEEP ONLY FREE CELLS WITH SUFFICIENT ROBOT CLEARANCE
        enough_clearance = clearance >= self.min_clearance
        safe = free & enough_clearance

        # CONVERT ROBOT MAP COORDINATES INTO GRID COORDINATES
        ox = origin[0]
        oy = origin[1]
        yaw = origin[2]
        dx = robot_xy[0] - ox
        dy = robot_xy[1] - oy

        robot_grid_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        robot_grid_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        col = int(math.floor(robot_grid_x / resolution))
        row = int(math.floor(robot_grid_y / resolution))

        # FIND THE NEAREST SAFE CELL TO THE ROBOT
        nearest = self._nearby_component(safe, col, row)
        reachable = np.zeros_like(free, dtype=bool)

        # DO NOT CHOOSE AN UNRELATED SAFE REGION FAR FROM THE ROBOT
        if nearest is None:
            return {'widest': None, 'bottleneck': None, 'clearance': clearance, 'reachable': reachable}
        nearest_distance = math.hypot(nearest[0] - col, nearest[1] - row) * resolution
        if nearest_distance > 1.5:
            return {'widest': None, 'bottleneck': None, 'clearance': clearance, 'reachable': reachable}

        # FIND THE CONNECTED SAFE COMPONENT CONTAINING THE ROBOT
        count, component = cv2.connectedComponents(safe.astype(np.uint8), connectivity=8)
        label = component[nearest[1], nearest[0]]
        if label == 0:
            return {'widest': None, 'bottleneck': None, 'clearance': clearance, 'reachable': reachable}
        reachable = component == label

        # FIND THE REACHABLE CELL WITH THE GREATEST CLEARANCE
        positions = np.nonzero(reachable)
        ys = positions[0]
        xs = positions[1]
        reachable_clearance = clearance[ys, xs]
        top = int(np.argmax(reachable_clearance))
        wr = int(ys[top])
        wc = int(xs[top])
        wx, wy = self.transform(wc, wr, resolution, origin)
        widest = Feature(wx, wy, float(clearance[wr, wc]), wc, wr)

        # FIND CENTRELINE CANDIDATES USING LOCAL CLEARANCE PEAKS
        k = max(3, int(round(1.0 / resolution)) | 1)
        clearance_float = clearance.astype(np.float32)
        horizontal_kernel = np.ones((1, k), np.uint8)
        vertical_kernel = np.ones((k, 1), np.uint8)
        horizontal = cv2.dilate(clearance_float, horizontal_kernel)
        vertical = cv2.dilate(clearance_float, vertical_kernel)
        tol = 0.25 * resolution
        ridge_horizontal = clearance >= horizontal - tol
        ridge_vertical = clearance >= vertical - tol
        ridge = ridge_horizontal | ridge_vertical

        # ONLY KEEP REACHABLE NARROWER REGIONS AS BOTTLENECK CANDIDATES
        upper = min(self.max_bottleneck, widest.clearance_m * 0.80)
        valid = reachable & ridge
        valid &= clearance >= self.min_clearance
        valid &= clearance <= upper

        # EXCLUDE FRONTIERS NEAR UNKNOWN MAP CELLS
        known_mask = (grid >= 0).astype(np.uint8)
        known_kernel = np.ones((5, 5), np.uint8)
        known_near = cv2.erode(known_mask, known_kernel) > 0
        valid &= known_near

        # SORT CANDIDATES FROM NARROWEST TO WIDEST
        candidates = np.argwhere(valid)
        if len(candidates):
            candidate_clearance = clearance[candidates[:, 0], candidates[:, 1]]
            ordering = np.argsort(candidate_clearance)
            candidates = candidates[ordering]

        # CHECK FOR SAFE PASSAGE ON BOTH SIDES OF THE CANDIDATE
        step = max(2, int(round(1.5 / resolution)))
        bottleneck = None
        for r, c in candidates[:4000]:
            along_x = c >= step and c + step < width and reachable[r, c-step] and reachable[r, c+step]
            along_y = r >= step and r + step < height and reachable[r-step, c] and reachable[r+step, c]
            if not (along_x or along_y):
                continue

            # CANDIDATE SHOULD BE A LOCAL MINIMUM ALONG A TRAVEL DIRECTION
            if along_x and (clearance[r, c] <= clearance[r, c-step] + tol and
                            clearance[r, c] <= clearance[r, c+step] + tol):
                pass
            elif along_y and (clearance[r, c] <= clearance[r-step, c] + tol and
                              clearance[r, c] <= clearance[r+step, c] + tol):
                pass
            else:
                continue

            # SAVE THE FIRST SUITABLE BOTTLENECK CANDIDATE
            bx, by = self.transform(int(c), int(r), resolution, origin)
            bottleneck = Feature(bx, by, float(clearance[r, c]), int(c), int(r))
            break

        return {'widest': widest, 'bottleneck': bottleneck,
                'clearance': clearance, 'reachable': reachable}
