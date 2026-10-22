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
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import TransformException
from cave_explorer.perception3_geometry import depth_point, transform_point, ArtefactTracks


class Perception3:
    def __init__(self, node):
        self.node = node
        self.depths = deque(maxlen=12)
        self.pending = deque(maxlen=12)
        self.tracker = ArtefactTracks()
        self.last_stamp = None
        self.last_warning = {}
        node.declare_parameter('p3_merge_radius', 0.75)
        node.declare_parameter('p3_sync_tolerance', 0.08)
        node.declare_parameter('p3_pending_timeout', 6.0) # AA ADDED 6 SECONDS TO ALLOW MORE TIME FOR DEPTH IMAGE AND TF DATA TO BE RECEIVED.
        node.declare_parameter('p3_hfov', 2.0944)
        self.tracker.radius = float(node.get_parameter('p3_merge_radius').value)
        if self.tracker.radius <= 0:
            raise ValueError('p3_merge_radius must be positive')
        self.sub = node.create_subscription(Image, 'camera/depth/image', self.depth_callback,
                                            qos_profile_sensor_data)
        self.timer = node.create_timer(0.05, self.process)
        self.marker_timer = node.create_timer(1.0, self.publish_markers)

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
        if self.depths and stamp < self.depths[-1][0]:
            self.reset()
        self.depths.append((stamp, depth))
        self.process()

    def submit(self, image_msg, detections):
        stamp = self.stamp(image_msg)
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.reset()
        if self.last_stamp == stamp:
            return
        self.last_stamp = stamp
        if not detections:
            return
        if image_msg.header.frame_id != 'camera_link':
            self.warn('rgb_frame', 'RGB must come from supplied Gazebo camera_link sensor')
            return
        self.pending.append((image_msg.header, image_msg.height, image_msg.width,
                             list(detections), time.monotonic()))
        self.process()

    def reset(self):
        self.depths.clear()
        self.pending.clear()
        self.tracker = ArtefactTracks(self.tracker.radius)
        self.last_stamp = None
        self.node.artifact_locations_ = []
        self.publish_markers()

    def process(self):
        # Queue briefly: depth or TF may arrive just after RGB.
        radius = float(self.node.get_parameter('p3_merge_radius').value)
        if radius > 0:
            self.tracker.radius = radius
        remaining = deque(maxlen=12)
        while self.pending:
            item = self.pending.popleft()
            header, height, width, detections, received = item
            stamp = header.stamp.sec + header.stamp.nanosec*1e-9
            if time.monotonic()-received > float(self.node.get_parameter('p3_pending_timeout').value): # AA CHANGED FROM 1.5 TO 6 SECONDS TO ALLOW MORE TIME FOR DEPTH IMAGE AND TF DATA TO BE RECEIVED.
                self.warn('timeout', 'Skipped observation: matching depth or timestamped TF unavailable')
                continue
            if not self.depths:
                remaining.append(item)
                continue
            ds, depth = min(self.depths, key=lambda pair: abs(pair[0]-stamp))
            tolerance = float(self.node.get_parameter('p3_sync_tolerance').value)
            if abs(ds-stamp) > tolerance:
                remaining.append(item)
                continue
            if depth.shape != (height, width) or (height, width) != (480, 720):
                self.warn('size', 'Expected aligned 720x480 RGB/depth; recalibrate for changed camera')
                continue
            try:
                tf = self.node.tf_buffer.lookup_transform('map', 'camera_link', Time.from_msg(header.stamp))
            except TransformException:
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
