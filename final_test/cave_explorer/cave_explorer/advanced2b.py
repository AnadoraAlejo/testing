# === END ASSIGNMENT 3 PROVENANCE ===
"""Advanced 2: online distance-transform geometry display and CSV evidence.

Standalone node; it only observes map/TF and never sends navigation goals.
"""
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from cave_explorer.advanced2a import GeometryAnalyser


# DEFINE ROS NODE 
class GeometryNode(Node):

    def __init__(self):
        super().__init__('cave_geometry')

        # SET PARAMETERS FOR SPEED, CLEARANCE AND RESULTS LOCATION
        self.declare_parameter('update_period_s', 2.0)
        self.declare_parameter('minimum_clearance_m', 0.75)
        self.declare_parameter('maximum_bottleneck_clearance_m', 3.0)
        self.declare_parameter('results_directory', '~/cave_results/geometry')

        # RETRIEVE MIN CLEARANCE AND MAX BOTTLENECK VALUES
        minimum_clearance = float(self.get_parameter('minimum_clearance_m').value)
        maximum_bottleneck = float(self.get_parameter('maximum_bottleneck_clearance_m').value)

        # ROBOT RADIUS = 0.5 M AND SAFETY MARGIN = MIN CLEARANCE - ROBOT RADIUS
        safety_margin = minimum_clearance - 0.5

        # USE ANALYSE TO IDENTIFY WIDEST AREA AND BOTTLENECK
        self.analyser = GeometryAnalyser(
            robot_radius=0.5,
            safety_margin=safety_margin,
            max_bottleneck_m=maximum_bottleneck
        )

        # SET UP TF TO LOCATE ROBOT POSITION ON THE MAP
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # MAP QUALITY OF SERVICE || KEEP LATEST MAP AND DELIVER RELIABLY
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # SUBSCRIBE TO MAP UPDATES
        self.create_subscription(OccupancyGrid, 'map', self.on_map, qos)

        # PUBLISH RVIZ MARKERS, CLEARANCE MAP AND STATUS RESULTS
        self.marker_publisher = self.create_publisher(
            MarkerArray, 'advanced2/markers', 10
        )
        self.clearance_publisher = self.create_publisher(
            OccupancyGrid, 'advanced2/clearance_grid', 10
        )
        self.status_publisher = self.create_publisher(
            String, 'advanced2/status', 10
        )

        # NO MAP IS STORED UNTIL THE FIRST MESSAGE IS RECEIVED
        self.latest_map = None

        # SET LOCATION TO SAVE GEOMETRY CSV RESULTS
        results_directory = self.get_parameter('results_directory').value
        self.output = Path(results_directory).expanduser()
        self.output.mkdir(parents=True, exist_ok=True)

        # CREATE CSV FILE NAME USING CURRENT DATE AND TIME
        date_time = time.strftime('%Y%m%d_%H%M%S')
        file_name = 'geometry_' + date_time + '.csv'
        self.csv_path = self.output / file_name

        # OPEN CSV AND WRITE COLUMN HEADERS
        self.csv_file = self.csv_path.open('w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'ros_stamp_s',
            'widest_x',
            'widest_y',
            'widest_clearance_m',
            'bottleneck_x',
            'bottleneck_y',
            'bottleneck_clearance_m'
        ])

        update_period = float(self.get_parameter('update_period_s').value)
        self.timer = self.create_timer(update_period, self.analyse)

        # DISPLAY THE CSV FILE LOCATION IN THE TERMINAL
        self.get_logger().info('Advanced 2 evidence: ' + str(self.csv_path))

    def on_map(self, message):

        # CHECK MAP MESSAGE IS IN THE MAP FRAME
        if message.header.frame_id != 'map':
            self.get_logger().warn('Expected /map in map frame')
            return

        # STORE THE MOST RECENT MAP
        self.latest_map = message


    # LOCATE ROBOT X AND Y POSITION USING TF
    def robot_xy(self):
        try:
            # TRANSFORM ROBOT BASE_LINK POSITION INTO MAP FRAME
            transform = self.buffer.lookup_transform('map', 'base_link', Time())

            # RETRIEVE ROBOT X AND Y COORDINATES
            robot_x = float(transform.transform.translation.x)
            robot_y = float(transform.transform.translation.y)

            return robot_x, robot_y

        except TransformException:
            # RETURN NOTHING IF MAP TO ROBOT TRANSFORM IS UNAVAILABLE
            return None


    # CONVERT IDENTIFIED FEATURE INTO A DICTIONARY FOR STATUS OUTPUT
    @staticmethod
    def feature_data(feature):

        # RETURN NOTHING
        if feature is None:
            return None

        # ROUND POSITION AND CLEARANCE TO 3 DECIMAL PLACES
        feature_x = round(feature.x, 3)
        feature_y = round(feature.y, 3)
        feature_clearance = round(feature.clearance_m, 3)

        return dict(
            x=feature_x,
            y=feature_y,
            clearance_m=feature_clearance
        )


    # ANALYSE LATEST MAP AND SAVE RESULTS
    def analyse(self):

        # RETRIEVE THE LATEST MAP
        msg = self.latest_map

        # DO NOT ANALYSE UNTIL THE MAP IS AVAILABLE
        if msg is None:
            return

        # RETRIEVE ROBOT POSITION IN MAP COORDINATES
        xy = self.robot_xy()

        # DO NOT ANALYSE UNTIL ROBOT POSITION IS AVAILABLE
        if xy is None:
            self.get_logger().warn('Advanced2 waiting for map -> base_link transform')
            return

        # RETRIEVE THE MAP ORIGIN ROTATION AS A QUATERNION
        q = msg.info.origin.orientation

        # CALCULATE MAP YAW (ROTATION) IN RADIANS
        # SEPARATE THE TOP AND BOTTOM OF THE ATAN2 EQUATION
        yaw_top = 2 * (q.w * q.z + q.x * q.y)
        yaw_bottom = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(yaw_top, yaw_bottom)

        # RETRIEVE THE MAP SIZE, RESOLUTION AND ORIGIN
        map_data = msg.data
        map_width = msg.info.width
        map_height = msg.info.height
        map_resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        map_origin = (origin_x, origin_y, yaw)

        try:
            # RUN ADVANCED 2 GEOMETRY CALCULATIONS
            results = self.analyser.analyse(
                map_data,
                map_width,
                map_height,
                map_resolution,
                map_origin,
                xy
            )

        except ValueError as error:
            # SKIP IF MAP DIMENSIONS OR DATA ARE INVALID
            self.get_logger().warn('Geometry skipped: ' + str(error))
            return

        # RETRIEVE WIDEST LOCATION AND BOTTLENECK CANDIDATE
        widest = results['widest']
        bottleneck = results['bottleneck']

        # DISPLAY RESULTS IN RVIZ
        self.publish_markers(widest, bottleneck)
        self.publish_clearance(msg, results)

        # CONVERT FEATURES INTO STATUS DATA AND PUBLISH AS JSON
        widest_status = self.feature_data(widest)
        bottleneck_status = self.feature_data(bottleneck)
        status = dict(widest=widest_status, bottleneck=bottleneck_status)
        status_json = json.dumps(status)
        self.status_publisher.publish(String(data=status_json))

        # CONVERT CURRENT ROS TIME FROM NANOSECONDS TO SECONDS
        current_nanoseconds = self.get_clock().now().nanoseconds
        current = current_nanoseconds * 1e-9

        # SAVE NAN IF WIDEST AREA IS NOT FOUND
        if widest is None:
            a = [float('nan')] * 3
        else:
            a = [widest.x, widest.y, widest.clearance_m]

        # SAVE NAN IF BOTTLENECK IS NOT FOUND
        if bottleneck is None:
            b = [float('nan')] * 3
        else:
            b = [bottleneck.x, bottleneck.y, bottleneck.clearance_m]

        # COMBINE BOTH FEATURE ARRAYS INTO ONE CSV ROW
        feature_values = a + b
        rounded_values = []

        # CONVERT EACH VALUE TO FLOAT AND ROUND TO 3 DECIMAL PLACES
        for value in feature_values:
            number = float(value)
            rounded_number = round(number, 3)
            rounded_values.append(rounded_number)

        # ADD THE TIME STAMP AT THE START OF THE CSV ROW
        csv_row = [round(current, 3)] + rounded_values
        self.writer.writerow(csv_row)

        # SAVE THE LATEST CSV DATA TO THE FILE
        self.csv_file.flush()


    # PUBLISH CLEARANCE MAP TO RVIZ
    def publish_clearance(self, msg, results):

        # OccupancyGrid values here are display-scaled clearance, NOT obstacles.
        # Add as a MAP display in RViz for a relative clearance heatmap.

        # CREATE A NEW OCCUPANCY GRID MESSAGE FOR DISPLAY
        result = OccupancyGrid()
        result.header = msg.header
        result.info = msg.info

        # RETRIEVE CLEARANCE ARRAY AND REACHABLE CELLS
        clearance = results['clearance']
        reachable = results['reachable']

        # SET ALL CELLS TO -1 (NO DISPLAY DATA)
        display = np.full(clearance.shape, -1, dtype=np.int8)

        # FIND MAXIMUM CLEARANCE ONLY IN REACHABLE CELLS
        if np.any(reachable):
            reachable_clearances = clearance[reachable]
            max_clearance = float(np.max(reachable_clearances))
        else:
            max_clearance = 0

        # CONVERT CLEARANCE VALUES INTO A SCALE FROM 0 TO 100
        if max_clearance > 0:
            reachable_clearances = clearance[reachable]
            clearance_ratio = reachable_clearances / max_clearance
            clearance_percentage = 100 * clearance_ratio
            rounded_clearance = np.round(clearance_percentage)
            limited_clearance = np.minimum(100, rounded_clearance)
            display[reachable] = limited_clearance.astype(np.int8)

        # FLATTEN 2D GRID INTO 1D ARRAY FOR OCCUPANCY GRID MESSAGE
        flat_display = display.flatten()
        result.data = flat_display.astype(int).tolist()

        # PUBLISH RELATIVE CLEARANCE MAP
        self.clearance_publisher.publish(result)


    # PUBLISH WIDEST LOCATION AND BOTTLENECK MARKERS IN RVIZ
    def publish_markers(self, widest, bottleneck):

        # CREATE MARKER ARRAY
        result = MarkerArray()

        # DELETE OLD MARKERS BEFORE ADDING CURRENT FEATURES
        deletion = Marker()
        deletion.action = Marker.DELETEALL
        result.markers.append(deletion)

        # SET MARKER LABEL, FEATURE, COLOUR AND SHAPE
        # WIDEST = BLUE SPHERE || BOTTLENECK = RED CUBE
        features = (
            ('WIDEST', widest, (0.0, 0.5, 1.0), Marker.SPHERE),
            ('BOTTLENECK', bottleneck, (1.0, 0.2, 0.0), Marker.CUBE)
        )

        for i, feature_details in enumerate(features):
            label = feature_details[0]
            feature = feature_details[1]
            rgb = feature_details[2]
            marker_type = feature_details[3]

            # SKIP MARKER IF THE FEATURE WAS NOT FOUND
            if feature is None:
                continue

            # MAKE A SHAPE MARKER AND A TEXT LABEL FOR EACH FEATURE
            for is_text in (False, True):
                marker = Marker()

                # SET MARKER POSITION IN THE MAP FRAME
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()

                # SET DIFFERENT NAMES AND IDENTIFIERS FOR SHAPE AND TEXT
                if is_text:
                    marker.ns = 'geometry_labels'
                else:
                    marker.ns = 'geometry_features'

                marker.id = 2 * i + int(is_text)

                # TEXT FACES THE VIEWER || FEATURE USES SPHERE OR CUBE
                if is_text:
                    marker.type = Marker.TEXT_VIEW_FACING
                else:
                    marker.type = marker_type

                marker.action = Marker.ADD

                # SET X, Y AND HEIGHT ABOVE THE MAP
                marker_x = feature.x
                marker_y = feature.y
                marker_z = 1.2 if is_text else 0.4
                marker.pose.position = Point(
                    x=marker_x,
                    y=marker_y,
                    z=marker_z
                )
                marker.pose.orientation.w = 1.0

                # SET MARKER DIMENSIONS
                marker.scale.x = 0.7
                marker.scale.y = 0.7
                marker.scale.z = 0.4 if not is_text else 0.45

                # SET RGB COLOUR AND FULL VISIBILITY
                marker.color.r = rgb[0]
                marker.color.g = rgb[1]
                marker.color.b = rgb[2]
                marker.color.a = 1.0

                # ADD FEATURE NAME AND CLEARANCE TO TEXT MARKER
                if is_text:
                    marker.text = f'{label}: {feature.clearance_m:.2f} m radius'

                # ADD THIS MARKER TO THE ARRAY
                result.markers.append(marker)

        # PUBLISH ALL MARKERS TO RVIZ
        self.marker_publisher.publish(result)


    # CLOSE CSV FILE WHEN NODE STOPS
    def close(self):
        self.csv_file.close()


# START ROS 2 NODE

def main(args=None):

    # INITIALISE ROS 2
    rclpy.init(args=args)

    # CREATE ADVANCED 2 GEOMETRY NODE
    node = GeometryNode()

    try:
        # KEEP NODE RUNNING TO RECEIVE MAP UPDATES
        rclpy.spin(node)

    except KeyboardInterrupt:
        # ALLOW NODE TO STOP USING CTRL + C
        pass

    finally:
        # CLOSE CSV AND DESTROY NODE BEFORE SHUTDOWN
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
