"""Planning 3 state machine, independent of ROS for event-sequence testing.

nav contract: busy, status, send(goal), cancel(). A cancelled goal remains
busy until its terminal result arrives. No overlapping navigation goals.
"""
import math


class BehaviourSwitcher:
    def __init__(self, nav, grid, emit=lambda event: None, chosen=('stop_sign',),
                 lost_timeout=8.0, approach_timeout=45.0, standoff=2.0):
        self.nav,self.grid,self.emit = nav,grid,emit
        self.chosen=set(chosen)
        self.lost_timeout=lost_timeout
        self.approach_timeout=approach_timeout
        self.standoff=standoff
        self.state='EXPLORE'
        self.target=None
        self.target_label=None
        self.target_point=None
        self.target_last=-math.inf
        self.attempt=0
        self.visited=[]
        self.abandoned=[]
        self.resume_goal=None
        self.explore_goal=None
        self.entered=0.0
        self.next_scan=0.0
        self.next_explore=0.0
        self.after_cancel=None
        self.approach_count=0
        self.latest_count=0
        self.has_seen_after_arrival=False

    def transition(self, state, now, reason):
        previous=self.state
        self.state,self.entered=state,now
        self.emit(dict(time=now,event='transition',old=previous,state=state,reason=reason,
                       target_id=self.target,attempt=self.attempt))

    def known(self, label, point):
        return any(label==name and math.dist(point[:2],p[:2])<1.0
                   for name,p in self.visited+self.abandoned)

    def observe(self, tracks):
        if self.target is None:
            return
        # Stable tracker id first, spatial association only if an id changed.
        matching=[t for t in tracks if t['label']==self.target_label and
                  (t['id']==self.target or math.dist(t['point'][:2],self.target_point[:2])<0.75)]
        if matching:
            t=max(matching,key=lambda t:t['last'])
            self.target_point=tuple(t['point'])
            self.target_last=t['last']
            self.latest_count=t['count']
            if self.state=='INSPECT' and t['last']>self.entered:
                self.has_seen_after_arrival=True

    def cancel_then(self, destination, now, reason):
        self.after_cancel=destination
        if self.nav.busy:
            self.nav.cancel()
            self.transition('CANCELLING',now,reason)
        else:
            self.transition(destination,now,reason)

    def fail_inspection(self, now, reason):
        if self.attempt==0:
            self.attempt=1
            self.cancel_then('RETRY',now,reason+'; retry once')
        else:
            self.abandoned.append((self.target_label,self.target_point))
            self.emit(dict(time=now,event='abandoned',label=self.target_label,point=self.target_point,reason=reason))
            self.cancel_then('RESUME',now,reason+'; attempt abandoned')

    def tick(self, now, robot, tracks):
        self.observe(tracks)
        if self.state=='HALTED':
            return
        if self.state=='CANCELLING':
            if not self.nav.busy:
                self.transition(self.after_cancel,now,'Previous navigation goal terminated')
            elif now-self.entered>15:
                self.transition('HALTED',now,'No terminal result after cancellation; no new goals sent')
            return
        if robot is None:
            # Loss of map pose should not create navigation commands.
            if self.nav.busy:
                self.cancel_then('RESUME',now,'Map pose unavailable')
            return
        if self.state=='RESUME':
            self.target=None
            self.target_label=None
            self.target_point=None
            self.explore_goal=None
            self.transition('EXPLORE',now,'Resume exploration')
            self.next_explore=now
        if self.state=='EXPLORE':
            available=[t for t in tracks if t['count']>=3 and t['label'] in self.chosen
                       and now-t['last']<2.0 and not self.known(t['label'],t['point'])]
            if available:
                target=min(available,key=lambda t:math.dist(robot[:2],t['point'][:2]))
                self.target=target['id']
                self.target_label=target['label']
                self.target_point=tuple(target['point'])
                self.target_last=target['last']
                self.latest_count=target['count']
                self.attempt=0
                self.resume_goal=self.explore_goal
                self.cancel_then('APPROACH_START',now,'New confirmed artefact')
                return
            if self.nav.busy:
                if now-self.entered>75:
                    if self.explore_goal:
                        self.grid.remember(self.explore_goal)
                    self.cancel_then('RESUME',now,'Exploration goal timeout')
                return
            if self.explore_goal is not None:
                self.grid.remember(self.explore_goal)
                self.explore_goal=None

                # AA ADD PROVIDE NAV 2 RECOVERY AFTER FAILED GOAL
                if self.nav.status in (
                    'ABORTED', 'REJECTED', 'FAILED',
                    'TRANSPORT_ERROR'
                ):
                    self.next_explore = max(
                        self.next_explore, now + 5.0
                    )
                elif self.nav.status == 'SUCCEEDED':
                    self.next_explore = max(
                        self.next_explore, now + 1.0
                    )

            if now<self.next_explore:
                return
            goal=self.resume_goal
            self.resume_goal=None
            if goal is None or not self.grid.safe_goal(goal,robot):
                goal=self.grid.explore(robot)
            if goal is None:
                self.next_explore=now+3
                return
            if self.nav.send(goal):
                self.explore_goal=goal
                self.entered=now
                self.emit(dict(time=now,event='explore_goal',goal=goal))
            else:
                self.next_explore=now+2
            return
        if self.state in ('APPROACH_START','RETRY'):
            goal=self.grid.approach(self.target_point,robot,self.attempt,self.standoff)
            if goal is None:
                self.fail_inspection(now,'No reachable clear viewpoint')
                return
            if self.nav.send(goal):
                self.approach_count=self.latest_count
                self.transition('APPROACH',now,'Approach viewpoint')
                self.emit(dict(time=now,event='approach_goal',goal=goal))
            elif now-self.entered>10:
                self.fail_inspection(now,'Navigation server unavailable')
            return
        if self.state=='APPROACH':
            # Allow a full reacquisition interval on the retry, not immediate failure.
            if now-self.entered>self.approach_timeout:
                self.fail_inspection(now,'Approach timeout')
            elif now-max(self.target_last,self.entered)>self.lost_timeout:
                self.fail_inspection(now,'Artefact lost')
            elif not self.nav.busy:
                if self.nav.status!='SUCCEEDED':
                    self.fail_inspection(now,'Navigation '+self.nav.status)
                elif math.dist(robot[:2],self.target_point[:2])>self.standoff+0.75:
                    self.fail_inspection(now,'Arrival too far from artefact')
                else:
                    self.has_seen_after_arrival=False
                    self.transition('INSPECT',now,'Reached viewpoint; wait for fresh visual confirmation')
            return
        if self.state=='INSPECT':
            if now-self.entered>=2 and self.has_seen_after_arrival and now-self.target_last<2:
                self.visited.append((self.target_label,self.target_point))
                self.emit(dict(time=now,event='visited',label=self.target_label,point=self.target_point))
                self.transition('RESUME',now,'Inspection visually confirmed')
            elif now-self.entered>self.lost_timeout:
                self.fail_inspection(now,'No fresh detection at viewpoint')
