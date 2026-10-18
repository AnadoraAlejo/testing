"""Perception 3: live RGB-D localisation, repeated sightings and RViz display.

Used inside CaveExplorer's existing single-threaded executor.
Detection contract: [(label, (x, y, width, height)), ...], original RGB pixels.
"""
from collections import deque
import time
import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time
from tf2_ros import TransformException
from cave_explorer.perception3_geometry import depth_point, transform_point, ArtefactTracks


class Perception3:
    def __init__(self, node):
        self.node = node
        self.depths = deque(maxlen=200)
        self.pending = deque(maxlen=120)
        self.tracker = ArtefactTracks()
        self.last_stamp = None
        self.last_processed_stamp = None
        self.last_clock = None
        self.last_warning = {}
        node.declare_parameter('p3_merge_radius', 0.75)
        node.declare_parameter('p3_sync_tolerance', 0.50)
        node.declare_parameter('p3_pending_timeout', 6.0)  # ROS/simulation seconds
        node.declare_parameter('p3_pending_wall_timeout', 600.0)  # stalled-clock watchdog
        node.declare_parameter('p3_hfov', 2.0944)
        self.tracker.radius = float(node.get_parameter('p3_merge_radius').value)
        if self.tracker.radius <= 0:
            raise ValueError('p3_merge_radius must be positive')
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            durability=DurabilityPolicy.VOLATILE)
        self.sub = node.create_subscription(Image, 'camera/depth/image', self.depth_callback,
                                            depth_qos)
        self.timer = node.create_timer(0.10, self.process)
        self.marker_timer = node.create_timer(1.0, self.publish_markers)
        node.get_logger().info(
            'Perception 3 team revision: aligned 480x320; depth history=200; '
            'depth subscription queue=200; exact image-time TF; '
            f"tolerance={node.get_parameter('p3_sync_tolerance').value}s")

    def warn(self, key, message):
        now = time.monotonic()
        if now-self.last_warning.get(key, -100) > 5:
            self.node.get_logger().warn('Perception 3: '+message)
            self.last_warning[key] = now

    @staticmethod
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9

    def depth_callback(self, msg):
        if msg.header.frame_id != 'camera_link':
            self.warn('frame', 'Expected supplied Gazebo camera_link frame; got '+msg.header.frame_id)
            return
        if msg.encoding not in ('32FC1', '16UC1'):
            self.warn('encoding', 'Unsupported depth encoding '+msg.encoding)
            return
        try:
            depth = self.node.cv_bridge_.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = np.asarray(depth, dtype=np.float32).copy()
            if msg.encoding == '16UC1':
                depth *= 0.001
            if depth.ndim != 2:
                raise ValueError('Depth must be a single-channel image')
        except Exception as error:
            self.warn('conversion', str(error))
            return
        stamp = self.stamp(msg)
        self.check_clock()
        # Late/out-of-order sensor delivery is not a simulation clock reset.
        frames = [(t, d) for t, d in self.depths if t != stamp]
        frames.append((stamp, depth))
        frames.sort(key=lambda pair: pair[0])
        self.depths = deque(frames[-200:], maxlen=200)

    def submit(self, image_msg, detections):
        stamp = self.stamp(image_msg)
        self.check_clock()
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return
        self.last_stamp = stamp
        if not detections:
            return
        if image_msg.header.frame_id != 'camera_link':
            self.warn('rgb_frame', 'RGB must come from supplied Gazebo camera_link sensor')
            return
        if len(self.pending) == self.pending.maxlen:
            self.warn('queue', 'Pending queue full; dropping oldest observation')
        self.pending.append(dict(
            header=image_msg.header, height=image_msg.height, width=image_msg.width,
            detections=list(detections), received_wall=time.monotonic(),
            received_ros=self.node.get_clock().now().nanoseconds/1e9,
            depth=None, reason='No depth message received'))

    def reset(self):
        self.depths.clear()
        self.pending.clear()
        self.tracker = ArtefactTracks(self.tracker.radius)
        self.last_stamp = None
        self.last_processed_stamp = None
        self.last_clock = None
        self.node.artifact_locations_ = []
        self.publish_markers()

    def check_clock(self):
        now = self.node.get_clock().now().nanoseconds/1e9
        if self.last_clock is not None and now < self.last_clock:
            self.reset()
        self.last_clock = now
        return now

    def process(self):
        # Queue briefly: depth or TF may arrive just after RGB.
        radius = float(self.node.get_parameter('p3_merge_radius').value)
        if radius > 0:
            self.tracker.radius = radius
        remaining = deque(maxlen=self.pending.maxlen)
        now_ros = self.check_clock()
        while self.pending:
            item = self.pending.popleft()
            header = item['header']
            height, width = item['height'], item['width']
            detections = item['detections']
            stamp = header.stamp.sec + header.stamp.nanosec*1e-9
            if self.last_processed_stamp is not None and stamp <= self.last_processed_stamp:
                # Do not let an old delayed observation move track freshness backwards.
                continue
            ros_age = now_ros-item['received_ros']
            wall_age = time.monotonic()-item['received_wall']
            if ros_age < 0:
                self.reset()
                return
            if (ros_age > float(self.node.get_parameter('p3_pending_timeout').value)
                    or wall_age > float(self.node.get_parameter('p3_pending_wall_timeout').value)):
                self.warn('timeout', f"Skipped observation at {stamp:.3f}: {item['reason']}; "
                          f"waited {ros_age:.2f}s ROS / {wall_age:.1f}s wall")
                continue
            # Retain the matched depth while waiting for timestamped TF. Otherwise
            # it can disappear from the rolling depth buffer before TF catches up.
            depth = item['depth']
            if depth is None:
                if not self.depths:
                    item['reason'] = 'No accepted depth on /camera/depth/image'
                    self.warn('depth_wait', item['reason'])
                    remaining.append(item)
                    continue
                ds, depth = min(self.depths, key=lambda pair: abs(pair[0]-stamp))
                tolerance = float(self.node.get_parameter('p3_sync_tolerance').value)
                if abs(ds-stamp) > tolerance:
                    item['reason'] = (f'RGB-depth mismatch: nearest delta={abs(ds-stamp):.3f}s '
                                      f'exceeds tolerance={tolerance:.3f}s; RGB={stamp:.3f}; '
                                      f'nearest_depth={ds:.3f}; depth_minus_rgb={ds-stamp:+.3f}s; '
                                      f'buffer=[{self.depths[0][0]:.3f}, {self.depths[-1][0]:.3f}]; '
                                      f'frames={len(self.depths)}; RGB_age={now_ros-stamp:.3f}s')
                    self.warn('sync_wait', item['reason'])
                    remaining.append(item)
                    continue
                if depth.shape != (height, width) or (height, width) != (320, 480):
                    self.warn('size', 'Expected aligned 480x320 RGB/depth; recalibrate for changed camera')
                    continue
                item['depth'] = depth
            try:
                tf = self.node.tf_buffer.lookup_transform('map', 'camera_link', Time.from_msg(header.stamp))
            except TransformException as error:
                item['reason'] = f'TF map <- camera_link at {stamp:.3f}: {error}'
                self.warn('tf_wait', item['reason'])
                remaining.append(item)
                continue
            trans, rot = tf.transform.translation, tf.transform.rotation
            observations = []
            for label, box in detections:
                if label == 'negative':
                    continue
                camera_point = depth_point(depth, box, float(self.node.get_parameter('p3_hfov').value))
                if camera_point is None:
                    continue
                point = transform_point(camera_point, np.array([trans.x, trans.y, trans.z]),
                                        [rot.x, rot.y, rot.z, rot.w])
                observations.append((str(label), point))
            self.last_processed_stamp = stamp
            if not observations:
                self.warn('depth_geometry',
                          'RGB/depth and TF matched, but no valid central-patch depth; '
                          'check depth validity and whether boxes cover the artefact')
            tracks = self.tracker.update(observations, stamp)
            self.node.artifact_locations_ = [Point(x=float(t['point'][0]), y=float(t['point'][1]),
                                                  z=float(t['point'][2])) for t in tracks]
            self.publish_markers()
        self.pending = remaining

    def publish_markers(self):
        array = MarkerArray()
        # This is the sole marker publisher on the starter's artefact topic.
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        colours = {'green_alien': (0.2, 1.0, 0.2), 'green_crystals': (0.0, 0.7, 0.5),
                   'white_sphere': (1.0, 1.0, 1.0), 'ice_formation': (0.3, 1.0, 1.0),
                   'blue_mushrooms': (0.2, 0.3, 1.0), 'stop_sign': (1.0, 0.2, 0.1)}
        for track in self.tracker.tracks:
            if track['count'] < 3:
                continue
            x, y, z = map(float, track['point'])
            for is_text in (False, True):
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.node.get_clock().now().to_msg()
                marker.ns = 'artefact_names' if is_text else 'artefact_positions'
                marker.id = track['id']
                marker.type = Marker.TEXT_VIEW_FACING if is_text else Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.pose.position = Point(x=x, y=y, z=z+(0.55 if is_text else 0.0))
                marker.scale.x = marker.scale.y = marker.scale.z = 0.35
                marker.color.r, marker.color.g, marker.color.b = colours.get(track['label'], (1.0, 0.7, 0.0))
                marker.color.a = 1.0
                marker.text = f"{track['label']} #{track['id']} ({x:.1f}, {y:.1f}, {z:.1f}) n={track['count']}"
                array.markers.append(marker)
        self.node.marker_pub_.publish(array)
