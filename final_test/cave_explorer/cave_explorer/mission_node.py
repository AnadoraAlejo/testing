"""Integrated runner for Planning 3 and Advanced 1, replacing starter autonomy.

Original simulator/navigation launches are reused. Do not run the old autonomy
launch alongside this node: both would otherwise command Nav2.
"""
import csv
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from cave_explorer.advanced1 import degrade, restore, annotate
from cave_explorer.perception2_detector import CompositeDetector
from cave_explorer.perception3_localisation import Perception3
from cave_explorer.planning_support import GridPlanner
from cave_explorer.planning3 import BehaviourSwitcher
from cave_explorer.nav_session import NavSession


class MissionNode(Node):
    def __init__(self):
        super().__init__('cave_mission')
        defaults={'degradation':'none','severity':0,'use_improved':False,
                  'planning_enabled':True,'chosen_types':['stop_sign'],
                  'lost_timeout':8.0,'approach_timeout':45.0,'standoff':2.0,
                  'output_directory':'~/cave_results','seed':42,
                  'computer_vision_model_filename':str(Path(get_package_share_directory('cave_explorer'))/'config/stop_data.xml')}
        for name,value in defaults.items():
            self.declare_parameter(name,value)
        self.cv_bridge_=CvBridge()
        self.tf_buffer=Buffer()
        self.tf_listener=TransformListener(self.tf_buffer,self)
        self.marker_pub_=self.create_publisher(MarkerArray,'marker_array_artifacts',10)
        self.artifact_locations_=[]
        self.perception3=Perception3(self)
        self.detector=CompositeDetector(self.value('computer_vision_model_filename'))
        chosen=self.value('chosen_types')
        if any(label not in self.detector.labels for label in chosen):
            raise ValueError('chosen_types must be supported by the detector; chosen classes must match detector.labels')
        self.grid=GridPlanner()
        self.nav=NavSession(self)
        self.output=Path(self.value('output_directory')).expanduser()/time.strftime('%Y%m%d_%H%M%S')
        self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'captures').mkdir()
        self.events=(self.output/'mission_events.jsonl').open('w')
        self.stats_file=(self.output/'live_counts.csv').open('w',newline='')
        self.stats=csv.writer(self.stats_file)
        self.stats.writerow(['stamp','kind','severity','clean_count','baseline_count','improved_count','pipeline_ms','selected'])
        self.switcher=BehaviourSwitcher(self.nav,self.grid,self.event,chosen,
                                       self.value('lost_timeout'),self.value('approach_timeout'),self.value('standoff'))
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub=self.create_subscription(OccupancyGrid,'map',self.map_callback,qos)
        self.image_sub=self.create_subscription(Image,'camera/image',self.image_callback,qos_profile_sensor_data)
        self.pubs={name:self.create_publisher(Image,'robust/'+name,1) for name in
                   ['original','degraded','restored','clean_detections','baseline_detections','improved_detections']}
        self.selected_pub=self.create_publisher(Image,'detections_image',1)
        self.status_pub=self.create_publisher(String,'planning3/status',10)
        self.plan_markers=self.create_publisher(MarkerArray,'planning3/markers',10)
        self.save_next=False
        self.save_sub=self.create_subscription(Empty,'robust/capture_save',self.request_capture,10)
        self.loop=self.create_timer(0.2,self.tick)
        self.display_timer=self.create_timer(1.0,self.display)
        self.frame=0
        self.last_tick=None
        self.enabled=bool(self.value('planning_enabled'))
        self.get_logger().info('Results: '+str(self.output))

    def value(self,name):
        return self.get_parameter(name).value

    def event(self,event):
        self.events.write(json.dumps(event)+'\n')
        self.events.flush()
        self.get_logger().info(json.dumps(event))

    def request_capture(self,_msg):
        self.save_next=True

    def map_callback(self,msg):
        if msg.header.frame_id!='map':
            self.get_logger().error('Expected occupancy grid frame map')
            return
        q=msg.info.origin.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        self.grid.update(msg.data,msg.info.width,msg.info.height,msg.info.resolution,
                         (msg.info.origin.position.x,msg.info.origin.position.y,yaw))

    def image_callback(self,msg):
        started=time.perf_counter()
        try:
            original=self.cv_bridge_.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            kind,level=self.value('degradation'),int(self.value('severity'))
            if kind=='none' or level==0:
                altered=original
                enhanced=original
                clean_boxes=self.detector.detect(original)
                baseline_boxes=clean_boxes
                improved_boxes=clean_boxes
            else:
                altered=degrade(original,kind,level,int(self.value('seed'))+self.frame)
                enhanced=restore(altered)
                clean_boxes=self.detector.detect(original)
                baseline_boxes=self.detector.detect(altered)
                improved_boxes=self.detector.detect(enhanced)
        except (ValueError,cv2.error,CvBridgeError) as error:
            self.get_logger().error('Perception frame skipped: '+str(error))
            return
        self.frame+=1
        # Only the selected degraded pipeline drives localisation/planning.
        use_improved=bool(self.value('use_improved'))
        selected=improved_boxes if use_improved else baseline_boxes
        self.perception3.submit(msg,[(d.label,d.box) for d in selected])
        if kind=='none' or level==0:
            shown=annotate(original,baseline_boxes,f'{kind} {level}: baseline')
            out=self.cv_bridge_.cv2_to_imgmsg(shown,encoding='bgr8')
            out.header=msg.header
            self.selected_pub.publish(out)
        else:
            images={'original':original,'degraded':altered,'restored':enhanced,
                    'clean_detections':annotate(original,clean_boxes,'Clean reference detections'),
                    'baseline_detections':annotate(altered,baseline_boxes,f'{kind} {level}: baseline'),
                    'improved_detections':annotate(enhanced,improved_boxes,f'{kind} {level}: restoration candidate')}
            for name,image in images.items():
                out=self.cv_bridge_.cv2_to_imgmsg(image,encoding='bgr8')
                out.header=msg.header
                self.pubs[name].publish(out)
                if name==('improved_detections' if use_improved else 'baseline_detections'):
                    self.selected_pub.publish(out)
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.stats.writerow([stamp,kind,level,len(clean_boxes),len(baseline_boxes),len(improved_boxes),
                             round((time.perf_counter()-started)*1000,3),'improved' if use_improved else 'baseline'])
        self.stats_file.flush()
        if self.save_next:
            filename=self.output/'captures'/f'frame_{self.frame:06d}.png'
            if not cv2.imwrite(str(filename),original):
                self.get_logger().error('Could not save capture')
            else:
                self.get_logger().info('Saved clean evaluation image: '+str(filename))
            self.save_next=False

    def robot_pose(self):
        try:
            tf=self.tf_buffer.lookup_transform('map','base_link',Time())
            q=tf.transform.rotation
            yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            stamp=tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
            if self.get_clock().now().nanoseconds/1e9-stamp>2:
                return None
            return (tf.transform.translation.x,tf.transform.translation.y,yaw)
        except TransformException:
            return None

    def tick(self):
        now=self.get_clock().now().nanoseconds/1e9
        if self.last_tick is not None and now<self.last_tick:
            self.nav.cancel()
            self.switcher.transition('HALTED',now,'Simulation clock reset: restart mission node')
        self.last_tick=now
        if not self.enabled:
            return
        tracks=[t for t in self.perception3.tracker.tracks if t['count']>=3]
        self.switcher.tick(now,self.robot_pose(),tracks)

    def display(self):
        s=self.switcher
        msg=String()
        msg.data=json.dumps(dict(state=s.state,target=s.target,attempt=s.attempt,
                                visited=len(s.visited),abandoned=len(s.abandoned),navigation=self.nav.status))
        self.status_pub.publish(msg)
        array=MarkerArray()
        clear=Marker(); clear.action=Marker.DELETEALL
        array.markers.append(clear)
        for t in self.perception3.tracker.tracks:
            if t['count']<3:
                continue
            point=t['point']; label=t['label']
            visited=any(label==name and math.dist(point[:2],p[:2])<1 for name,p in s.visited)
            abandoned=any(label==name and math.dist(point[:2],p[:2])<1 for name,p in s.abandoned)
            status='VISITED' if visited else 'ABANDONED' if abandoned else 'TARGET' if t['id']==s.target else 'UNVISITED'
            marker=Marker()
            marker.header.frame_id='map'; marker.header.stamp=self.get_clock().now().to_msg()
            marker.ns='inspection_status'; marker.id=t['id']; marker.type=Marker.TEXT_VIEW_FACING
            marker.pose.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2])+1.0)
            marker.pose.orientation.w=1.0; marker.scale.z=0.35
            marker.color.r,marker.color.g,marker.color.b = {'VISITED':(0.,1.,0.),'ABANDONED':(1.,0.,0.),
                                                        'TARGET':(1.,0.6,0.),'UNVISITED':(0.,0.7,1.)}[status]
            marker.color.a=1.0; marker.text=status+' '+label
            array.markers.append(marker)
        self.plan_markers.publish(array)

    def close_files(self):
        self.events.close()
        self.stats_file.close()


def main(args=None):
    rclpy.init(args=args)
    node=MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.enabled=False
        node.loop.cancel()
        if rclpy.ok() and node.nav.busy:
            node.nav.cancel()
            deadline=time.monotonic()+3
            while node.nav.busy and time.monotonic()<deadline and rclpy.ok():
                rclpy.spin_once(node,timeout_sec=0.1)
        node.close_files()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
