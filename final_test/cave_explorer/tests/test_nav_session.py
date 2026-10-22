"""Action-order tests with ROS message/API doubles; not a ROS integration test."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch


class Future:
    def __init__(self): self.callback=None; self.value=None
    def add_done_callback(self,callback): self.callback=callback
    def result(self): return self.value
    def finish(self,value):
        self.value=value
        if self.callback: self.callback(self)


class Handle:
    def __init__(self):
        self.accepted=True; self.result_future=Future(); self.cancel_future=Future(); self.cancel_count=0
    def get_result_async(self): return self.result_future
    def cancel_goal_async(self): self.cancel_count+=1; return self.cancel_future


class Client:
    def __init__(self,*args): self.future=Future(); self.sent=0
    def server_is_ready(self): return True
    def send_goal_async(self,request): self.sent+=1; return self.future


class Pose:
    def __init__(self):
        self.header=NS(frame_id='',stamp=None)
        self.pose=NS(position=NS(x=0,y=0),orientation=NS(z=0,w=1))


def load_session():
    modules={}
    for name in ['action_msgs.msg','geometry_msgs.msg','nav2_msgs.action','rclpy.action']:
        modules[name]=ModuleType(name)
    modules['action_msgs.msg'].GoalStatus=NS(STATUS_SUCCEEDED=4,STATUS_CANCELED=5,STATUS_ABORTED=6)
    modules['geometry_msgs.msg'].PoseStamped=Pose
    modules['nav2_msgs.action'].NavigateToPose=NS(Goal=lambda:NS(pose=None))
    modules['rclpy.action'].ActionClient=Client
    spec=importlib.util.spec_from_file_location('tested_nav',Path(__file__).parents[1]/'cave_explorer/nav_session.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,modules): spec.loader.exec_module(module)
    return module.NavSession


class ActionRaceTests(unittest.TestCase):
    def setUp(self):
        logger=NS(error=lambda text:None,warn=lambda text:None)
        node=NS(create_publisher=lambda *args:NS(publish=lambda msg:None),get_logger=lambda:logger,
                get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:None)))
        self.nav=load_session()(node)
    def test_cancel_before_acceptance_waits_for_terminal_result(self):
        self.assertTrue(self.nav.send((1,2,0)))
        self.nav.cancel()
        self.assertTrue(self.nav.busy)
        self.assertFalse(self.nav.send((3,4,0)))
        handle=Handle(); self.nav.client.future.finish(handle)
        self.assertEqual(handle.cancel_count,1)
        handle.cancel_future.finish(NS(goals_canceling=[1]))
        self.assertTrue(self.nav.busy)
        handle.result_future.finish(NS(status=5))
        self.assertFalse(self.nav.busy)
        self.assertEqual(self.nav.status,'CANCELED')
    def test_rejected_goal_clears_busy(self):
        self.nav.send((1,2,0)); handle=Handle(); handle.accepted=False
        self.nav.client.future.finish(handle)
        self.assertFalse(self.nav.busy)
        self.assertEqual(self.nav.status,'REJECTED')
    def test_cancel_rejected_does_not_allow_overlap(self):
        self.nav.send((1,2,0)); handle=Handle(); self.nav.client.future.finish(handle)
        self.nav.cancel(); handle.cancel_future.finish(NS(goals_canceling=[]))
        self.assertTrue(self.nav.busy)
        self.assertFalse(self.nav.send((3,4,0)))
        handle.result_future.finish(NS(status=4))
        self.assertFalse(self.nav.busy)
        self.assertEqual(self.nav.status,'SUCCEEDED')
