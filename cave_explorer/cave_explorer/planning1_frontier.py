"""Planning 1: score connected frontier clusters by information gain and travel.

"""
import math
import cv2
import numpy as np


# SELECT FRONTIER FOR EXPLORATION 
def select_frontier_goal(planner, robot, min_cluster_cells=6):

    # CHECK IF MAP RECEIVED, IF NO MAP NO GOAL 
    if planner.grid is None:
        return None

    # ROBOT POSITION -> MAP COORD
    robot_c, robot_r = planner.cell(robot)

    # CHECK IN MAP
    if not planner.inside(robot_c, robot_r):
        return None

    # FIND ROBOT
    component = planner.components[robot_r, robot_c]

    # IF NOT INSAFE AREA
    if component == 0:
        return None

    # EXPLORE IN SAFE AREA ONLY
    reachable = (planner.components == component)

    # UNKNWON -> 8 BUT MASK
    unknown_cells = planner.grid < 0
    unknown = unknown_cells.astype(np.uint8)

    radius_in_cells = 1.5 / planner.res
    rounded_radius = int(round(radius_in_cells))
    kernel_width = rounded_radius * 2 + 1
    radius_cells = max(3, kernel_width)

    kernel_shape = (radius_cells, radius_cells)
    kernel = np.ones(kernel_shape, np.uint8)
    expanded_unknown = cv2.dilate(unknown, kernel)
    adjacent_unknown = expanded_unknown != 0

    frontier_cells = reachable & adjacent_unknown
    frontier = frontier_cells.astype(np.uint8)

    cluster_results = cv2.connectedComponentsWithStats(frontier, connectivity=8)
    count = cluster_results[0]
    labels = cluster_results[1]
    stats = cluster_results[2]
    centers = cluster_results[3]

    # ARRAY OF POSSIBILITIES
    possibilities = []

    for index in range(1, count):

        # HOW MANY CELL IN AREA
        cluster_stats = stats[index]
        area = int(cluster_stats[cv2.CC_STAT_AREA])

        # SMALL ONE IGNORE
        if area < min_cluster_cells:
            continue

        # IDENTIFY ROW AND COL
        cluster_cells = labels == index
        positions = np.where(cluster_cells)
        rows = positions[0]
        cols = positions[1]

        # FIND CELLS TO CENTRE CLUSTER
        center = centers[index]
        cx = center[0]
        cy = center[1]

        col_difference = cols - cx
        row_difference = rows - cy
        col_squared = col_difference ** 2
        row_squared = row_difference ** 2
        distance_squared = col_squared + row_squared
        ordering = np.argsort(distance_squared)

        # 30 CELLS CHECK
        number_to_check = min(30, len(ordering))
        selected_cells = ordering[:number_to_check]
        for i in selected_cells:

            # GRID CELL -> MAP COORD
            selected_col = int(cols[i])
            selected_row = int(rows[i])
            x, y = planner.world(selected_col, selected_row)

            # STRAIGHT LINE GOAL TO ROBOT
            difference_x = x - robot[0]
            difference_y = y - robot[1]
            distance = math.hypot(difference_x, difference_y)

            # NO GOAL TOO CLOSE
            if distance < 1.2:
                continue
            if any(math.hypot(x-oldx, y-oldy) < 1.5 for oldx, oldy in planner.history[-100:]):
                continue
            # 
            score = 0.65 * math.log1p(area) - 0.11 * distance
            heading = math.atan2(y-robot[1], x-robot[0])
            goal = (x, y, heading)
            possibility = (score, goal)
            possibilities.append(possibility)
            break
    if possibilities:
        best_possibility = max(possibilities, key=lambda item: item[0])
        return best_possibility[1]

    return None
