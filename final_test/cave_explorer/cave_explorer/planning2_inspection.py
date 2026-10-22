# PLANNING 2
# this file only works out where to stand so we can look at an artefact up close
# it doesnt drive the rover and it doesnt explore the cave

import math


def same_spot(a, b, radius):
    # two points count as the same artefact if theyre close
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= radius


def worth_inspecting(points, done, radius=1.5):
    # the starter saves the robot spot with z = 1
    # thats not the artefact so those get skipped
    # real points from perception get kept, duplicates get dropped
    kept = []
    for point in points:
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        if abs(z - 1.0) < 1e-3:
            continue
        if any(same_spot((x, y), old, radius) for old in done):
            continue
        if any(same_spot((x, y), old, radius) for old in kept):
            continue
        kept.append((x, y))
    return kept


def _blocked(x, y, origin_x, origin_y, resolution, width, height, data, radius):
    # unknown cells and spots off the map are left for nav2
    # a known wall gets skipped so we dont pick a goal inside it
    if data is None or resolution <= 0.0:
        return False
    for dx in (-radius, 0.0, radius):
        for dy in (-radius, 0.0, radius):
            mx = int((x + dx - origin_x) / resolution)
            my = int((y + dy - origin_y) / resolution)
            if mx < 0 or my < 0 or mx >= width or my >= height:
                continue
            if int(data[my * width + mx]) >= 50:
                return True
    return False


def inspection_viewpoints(target_x, target_y, robot_x, robot_y, distance, map_info=None):
    # stand on the side we are already coming from, then a bit left and right
    # face the artefact, and stay back so we dont drive into it
    dx = robot_x - target_x
    dy = robot_y - target_y
    if math.hypot(dx, dy) < 1e-3:
        base = 0.0
    else:
        base = math.atan2(dy, dx)

    views = []
    for extra in (0.0, 0.7, -0.7):
        ang = base + extra
        x = target_x + distance * math.cos(ang)
        y = target_y + distance * math.sin(ang)
        theta = math.atan2(target_y - y, target_x - x)
        if theta < 0.0:
            theta = theta + 2.0 * math.pi
        if map_info is not None and _blocked(x, y, *map_info):
            continue
        views.append((x, y, theta))
    return views
# PLANNING 2 END
