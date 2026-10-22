#!/usr/bin/env python3
"""Planning 1: autonomous frontier exploration for the Mars cave.

Run alongside cave_explorer_startup.launch.py and
cave_explorer_navigation.launch.py (not autonomy.launch.py).

The robot obtains its pose from TF (map -> base_link), discovers frontiers
between known free cells and unknown cells in /map, selects reachable frontier
goals, and sends NavigateToPose actions to Nav2. Visualisation markers appear on
/planning1/markers. This is a self-contained demo, not a replacement for Nav2.
"""

import math
from collections import deque

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('planning1_frontier_explorer')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('min_goal_distance', 1.0)
        self.declare_parameter('max_goal_distance', 15.0)
        self.declare_parameter('min_frontier_cells', 4)
        self.declare_parameter('free_threshold', 20)
        self.declare_parameter('robot_clearance_m', 0.28)
        self.declare_parameter('goal_timeout_s', 120.0)
        self.declare_parameter('retry_radius_m', 1.5)
        self.declare_parameter('planning_interval_s', 4.0)
        self.declare_parameter('max_search_cells', 200000)

        self.map_msg = None
        self.goal_active = False
        self.goal_sending = False
        self.goal_handle = None
        self.goal_xy = None
        self.goal_start_ns = None
        self.cancelling = False
        self.failed_positions = []
        self.visited_positions = []
        self.last_notice_ns = 0

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(
            OccupancyGrid, self.get_parameter('map_topic').value,
            self.on_map, qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/planning1/markers', 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(float(self.get_parameter('planning_interval_s').value), self.plan)
        self.get_logger().info('Planning 1 frontier explorer started. Waiting for /map, TF and Nav2.')

    def on_map(self, msg):
        self.map_msg = msg

    def note(self, message):
        now = self.get_clock().now().nanoseconds
        if now - self.last_notice_ns > 10_000_000_000:
            self.get_logger().info(message)
            self.last_notice_ns = now

    def robot_xy(self, frame):
        transform = self.tf_buffer.lookup_transform(
            frame, self.get_parameter('robot_frame').value, rclpy.time.Time())
        t = transform.transform.translation
        return t.x, t.y

    @staticmethod
    def to_world(i, width, res, ox, oy, yaw):
        gx = (i % width + 0.5) * res
        gy = (i // width + 0.5) * res
        return ox + math.cos(yaw) * gx - math.sin(yaw) * gy, oy + math.sin(yaw) * gx + math.cos(yaw) * gy

    @staticmethod
    def to_grid(x, y, width, height, res, ox, oy, yaw):
        dx, dy = x - ox, y - oy
        gx = (math.cos(yaw) * dx + math.sin(yaw) * dy) / res
        gy = (-math.sin(yaw) * dx + math.cos(yaw) * dy) / res
        col, row = int(math.floor(gx)), int(math.floor(gy))
        if 0 <= col < width and 0 <= row < height:
            return row * width + col
        return None

    def plan(self):
        if self.map_msg is None:
            self.note('Waiting for an occupancy grid on /map. Start the navigation launch file.')
            return
        if self.goal_active or self.goal_sending:
            if (self.goal_active and not self.cancelling and self.goal_start_ns is not None
                    and self.get_clock().now().nanoseconds - self.goal_start_ns
                    > float(self.get_parameter('goal_timeout_s').value) * 1e9):
                self.get_logger().warning('Navigation timeout: cancelling goal.')
                self.cancelling = True
                if self.goal_handle is not None:
                    self.goal_handle.cancel_goal_async()
            return
        if not self.nav_client.server_is_ready():
            self.note('Waiting for Nav2 navigate_to_pose action server.')
            return

        msg = self.map_msg
        width, height = msg.info.width, msg.info.height
        if width == 0 or height == 0 or len(msg.data) != width * height:
            return
        origin = msg.info.origin
        q = origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        res = msg.info.resolution
        ox, oy = origin.position.x, origin.position.y
        frame = msg.header.frame_id or 'map'
        try:
            rx, ry = self.robot_xy(frame)
        except TransformException as exc:
            self.note('Waiting for map-to-robot TF: ' + str(exc))
            return
        start = self.to_grid(rx, ry, width, height, res, ox, oy, yaw)
        if start is None:
            self.note('Robot is outside the current map.')
            return

        cells = msg.data
        free_limit = int(self.get_parameter('free_threshold').value)
        free = lambda index: 0 <= cells[index] < free_limit
        if not free(start):
            self.note('Robot cell is not yet mapped as free.')
            return
        clearance = max(0, int(math.ceil(
            float(self.get_parameter('robot_clearance_m').value) / res)))

        def neighbours(i):
            row, col = divmod(i, width)
            if col > 0:
                yield i - 1
            if col + 1 < width:
                yield i + 1
            if row > 0:
                yield i - width
            if row + 1 < height:
                yield i + width

        def clear(i):
            if not free(i):
                return False
            row, col = divmod(i, width)
            # Avoid planning goals in obstacle-adjacent cells. Unknown cells
            # are allowed here because a frontier borders unexplored space.
            for dy in range(-clearance, clearance + 1):
                rr = row + dy
                if rr < 0 or rr >= height:
                    return False
                for dx in range(-clearance, clearance + 1):
                    cc = col + dx
                    if cc < 0 or cc >= width:
                        return False
                    if cells[rr * width + cc] >= free_limit:
                        return False
            return True

        # BFS estimates travel distance through known free map cells.
        # Only connected, currently reachable map cells are candidates.
        distances = {start: 0}
        queue = deque([start])
        limit = int(self.get_parameter('max_search_cells').value)
        while queue and len(distances) < limit:
            current = queue.popleft()
            for nb in neighbours(current):
                if nb not in distances and free(nb):
                    distances[nb] = distances[current] + 1
                    queue.append(nb)

        frontiers = {i for i in distances if any(cells[j] == -1 for j in neighbours(i))}
        seen = set()
        candidates = []
        min_size = int(self.get_parameter('min_frontier_cells').value)
        min_dist = float(self.get_parameter('min_goal_distance').value)
        max_dist = float(self.get_parameter('max_goal_distance').value)
        retry_radius = float(self.get_parameter('retry_radius_m').value)

        for root in frontiers:
            if root in seen:
                continue
            region = []
            pending = [root]
            seen.add(root)
            while pending:
                cell = pending.pop()
                region.append(cell)
                for nb in neighbours(cell):
                    if nb in frontiers and nb not in seen:
                        seen.add(nb)
                        pending.append(nb)
            if len(region) < min_size:
                continue
            # Choose a safely reachable cell near the frontier.
            valid = [i for i in region if clear(i)
                     and min_dist <= distances[i] * res <= max_dist]
            if not valid:
                continue
            choice = min(valid, key=lambda i: distances[i])
            x, y = self.to_world(choice, width, res, ox, oy, yaw)
            if any(math.hypot(x - fx, y - fy) < retry_radius for fx, fy in self.failed_positions):
                continue
            if any(math.hypot(x - vx, y - vy) < retry_radius for vx, vy in self.visited_positions):
                continue
            # Reward larger unexplored boundaries while penalising travel.
            score = len(region) / (1.0 + distances[choice] * res)
            candidates.append((score, choice, x, y, len(region)))

        if not candidates:
            self.note('No reachable unvisited frontier goals available yet.')
            self.publish_markers(frame, None)
            return
        best = max(candidates, key=lambda c: c[0])
        _, _, gx, gy, size = best
        self.publish_markers(frame, (gx, gy))
        self.send_goal(frame, gx, gy, math.atan2(gy - ry, gx - rx))
        self.get_logger().info(f'Selected frontier: x={gx:.2f}, y={gy:.2f}, size={size}, candidates={len(candidates)}')

    def publish_markers(self, frame, xy):
        arr = MarkerArray()
        delete = Marker()
        delete.header.frame_id = frame
        delete.action = Marker.DELETEALL
        arr.markers.append(delete)
        if xy is not None:
            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'planning1_frontier_goal'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = xy
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.45
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 0.2, 1.0
            arr.markers.append(marker)
        self.marker_pub.publish(arr)

    def send_goal(self, frame, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.goal_sending = True
        self.goal_xy = (x, y)
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        self.goal_sending = False
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error('Goal submission failed: ' + str(exc))
            self.blacklist_current()
            return
        if not handle.accepted:
            self.get_logger().warning('Nav2 rejected frontier goal.')
            self.blacklist_current()
            return
        self.goal_handle = handle
        self.goal_active = True
        self.cancelling = False
        self.goal_start_ns = self.get_clock().now().nanoseconds
        handle.get_result_async().add_done_callback(self.goal_result)
        self.get_logger().info('Nav2 accepted the frontier goal.')

    def blacklist_current(self):
        if self.goal_xy is not None:
            self.failed_positions.append(self.goal_xy)
        self.goal_xy = None
        self.goal_active = False
        self.goal_handle = None
        self.cancelling = False
        self.goal_start_ns = None

    def goal_result(self, future):
        try:
            result = future.result()
            success = result.status == GoalStatus.STATUS_SUCCEEDED
        except Exception as exc:
            self.get_logger().error('Navigation result failed: ' + str(exc))
            success = False
        if success:
            self.get_logger().info('Frontier reached; selecting next unexplored area.')
            if self.goal_xy is not None:
                self.visited_positions.append(self.goal_xy)
            self.goal_xy = None
            self.goal_active = False
            self.goal_handle = None
            self.goal_start_ns = None
            self.cancelling = False
        else:
            self.get_logger().warning('Could not reach frontier; avoiding that location.')
            self.blacklist_current()


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
