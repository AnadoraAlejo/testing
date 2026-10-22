import unittest
from cave_explorer.planning3 import BehaviourSwitcher


class FakeNav:
    def __init__(self):
        self.busy=False; self.status='IDLE'; self.sent=[]; self.cancels=0
    def send(self,goal):
        if self.busy:
            raise AssertionError('Overlapping goal')
        self.sent.append(goal); self.busy=True; self.status='ACTIVE'; return True
    def cancel(self):
        self.cancels+=1 # Still busy until result received.
    def finish(self,status='SUCCEEDED'):
        self.busy=False; self.status=status


class FakeGrid:
    def __init__(self): self.history=[]; self.valid=True
    def safe_goal(self,goal,robot): return self.valid
    def explore(self,robot): return (10.,0.,0.)
    def approach(self,target,robot,attempt,standoff): return (3.,0.,0.) if self.valid else None
    def remember(self,goal): self.history.append(goal)


def track(stamp=0,count=3,identity=7,label='stop_sign',point=(5.,0.,1.)):
    return dict(id=identity,label=label,point=point,count=count,last=stamp)


class SwitchingTests(unittest.TestCase):
    def setUp(self):
        self.nav=FakeNav(); self.grid=FakeGrid(); self.events=[]
        self.fsm=BehaviourSwitcher(self.nav,self.grid,self.events.append)
    def begin_approach(self):
        self.fsm.tick(0,(0,0,0),[track()])
        self.fsm.tick(0.2,(0,0,0),[track()])
        self.assertEqual(self.fsm.state,'APPROACH')
    def test_cancel_waits_for_terminal_result(self):
        self.fsm.tick(0,(0,0,0),[])
        self.fsm.tick(1,(0,0,0),[track(1)])
        self.assertEqual(self.fsm.state,'CANCELLING')
        self.fsm.tick(2,(0,0,0),[track(2)])
        self.assertEqual(len(self.nav.sent),1)
        self.nav.finish('CANCELED')
        self.fsm.tick(3,(0,0,0),[track(3)])
        self.fsm.tick(3.2,(0,0,0),[track(3)])
        self.assertEqual(self.fsm.state,'APPROACH')
        self.assertEqual(len(self.nav.sent),2)
    def test_success_requires_fresh_image_and_resumes(self):
        self.begin_approach(); self.nav.finish()
        self.fsm.tick(1,(3,0,0),[track(0)])
        self.fsm.tick(3.1,(3,0,0),[track(0)])
        self.assertFalse(self.fsm.visited)
        self.fsm.tick(3.2,(3,0,0),[track(3.2,5)])
        self.assertEqual(len(self.fsm.visited),1)
        self.fsm.tick(3.4,(3,0,0),[track(3.4,6,identity=99)])
        self.assertEqual(self.fsm.state,'EXPLORE')
        self.assertEqual(len(self.nav.sent),2)
    def test_lost_retry_once_then_abandon(self):
        self.begin_approach()
        self.fsm.tick(9,(0,0,0),[track(0)])
        self.assertEqual(self.fsm.attempt,1)
        self.assertEqual(self.fsm.state,'CANCELLING')
        self.nav.finish('CANCELED')
        self.fsm.tick(9.1,(0,0,0),[])
        self.fsm.tick(9.2,(0,0,0),[])
        self.assertEqual(self.fsm.state,'APPROACH')
        self.fsm.tick(10,(0,0,0),[]) # retry has its own grace period
        self.assertEqual(self.fsm.state,'APPROACH')
        self.fsm.tick(18,(0,0,0),[])
        self.assertEqual(len(self.fsm.abandoned),1)
        self.nav.finish('CANCELED')
        self.fsm.tick(18.1,(0,0,0),[])
        self.fsm.tick(18.2,(0,0,0),[track(18.2,10)])
        self.assertEqual(self.fsm.state,'EXPLORE')
        self.assertEqual(len(self.nav.sent),3) # two approaches + resumed exploration
    def test_aborted_not_visited(self):
        self.begin_approach(); self.nav.finish('ABORTED')
        self.fsm.tick(1,(3,0,0),[track(1)])
        self.assertEqual(self.fsm.attempt,1)
        self.assertFalse(self.fsm.visited)
    def test_cancel_timeout_halts(self):
        self.fsm.tick(0,(0,0,0),[])
        self.fsm.tick(1,(0,0,0),[track(1)])
        self.fsm.tick(17,(0,0,0),[track(17)])
        self.assertEqual(self.fsm.state,'HALTED')
        self.assertEqual(len(self.nav.sent),1)
    def test_unknown_class_does_not_interrupt(self):
        self.fsm.tick(0,(0,0,0),[track(label='green_alien')])
        self.assertEqual(self.fsm.state,'EXPLORE')
        self.assertIsNone(self.fsm.target)
    def test_missing_pose_cancels(self):
        self.fsm.tick(0,(0,0,0),[])
        self.fsm.tick(1,None,[])
        self.assertEqual(self.nav.cancels,1)
        self.assertEqual(self.fsm.state,'CANCELLING')
    def test_no_viewpoint_retries_and_abandons(self):
        self.grid.valid=False
        self.fsm.tick(0,(0,0,0),[track()])
        self.fsm.tick(.2,(0,0,0),[track()])
        self.fsm.tick(.4,(0,0,0),[track()])
        self.assertEqual(len(self.fsm.abandoned),1)
        self.assertFalse(self.nav.sent)
