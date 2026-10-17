"""One asynchronous Nav2 action at a time, including cancellation races."""
import math
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


class NavSession:
    def __init__(self, node):
        self.node=node
        self.client=ActionClient(node,NavigateToPose,'navigate_to_pose')
        self.busy=False
        self.status='IDLE'
        self.handle=None
        self.cancel_requested=False
        self.cancel_sent=False
        self.goal_pub=node.create_publisher(PoseStamped,'goal_pose',10)

    def send(self, goal):
        if self.busy or not self.client.server_is_ready():
            return False
        pose=PoseStamped()
        pose.header.frame_id='map'
        pose.header.stamp=self.node.get_clock().now().to_msg()
        pose.pose.position.x,pose.pose.position.y=map(float,goal[:2])
        pose.pose.orientation.z=math.sin(goal[2]/2)
        pose.pose.orientation.w=math.cos(goal[2]/2)
        request=NavigateToPose.Goal()
        request.pose=pose
        self.busy=True
        self.status='PENDING'
        self.handle=None
        self.cancel_requested=False
        self.cancel_sent=False
        try:
            future=self.client.send_goal_async(request)
            future.add_done_callback(self.accepted)
            self.goal_pub.publish(pose)
        except Exception as error:
            # Uncertain transport state: remain busy, never overlap another goal.
            self.status='TRANSPORT_ERROR'
            self.node.get_logger().error(str(error))
        return True

    def accepted(self, future):
        try:
            self.handle=future.result()
            if not self.handle.accepted:
                self.busy=False
                self.status='REJECTED'
                return
            self.status='ACTIVE'
            self.handle.get_result_async().add_done_callback(self.finished)
            if self.cancel_requested:
                self.cancel()
        except Exception as error:
            self.status='TRANSPORT_ERROR'
            self.node.get_logger().error(str(error))

    def cancel(self):
        self.cancel_requested=True
        if self.handle is not None and not self.cancel_sent:
            self.cancel_sent=True
            self.status='CANCELLING'
            try:
                self.handle.cancel_goal_async().add_done_callback(self.cancel_reply)
            except Exception as error:
                self.node.get_logger().error('Cancel request failed: '+str(error))

    def cancel_reply(self, future):
        try:
            if not future.result().goals_canceling:
                self.node.get_logger().warn('Nav2 did not accept cancellation; waiting for terminal result')
        except Exception as error:
            self.node.get_logger().error('Cancel reply failed: '+str(error))
        # Acknowledgement is not a terminal result: do not clear busy here.

    def finished(self, future):
        try:
            status=future.result().status
            self.status={GoalStatus.STATUS_SUCCEEDED:'SUCCEEDED',GoalStatus.STATUS_CANCELED:'CANCELED',
                         GoalStatus.STATUS_ABORTED:'ABORTED'}.get(status,'FAILED')
            self.busy=False
            self.handle=None
        except Exception as error:
            self.status='TRANSPORT_ERROR'
            self.node.get_logger().error('Result failed: '+str(error))
