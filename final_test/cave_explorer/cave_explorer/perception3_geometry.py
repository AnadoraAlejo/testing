"""ROS-independent geometry for the supplied Gazebo RGB-D camera."""
import math
import numpy as np


def depth_point(depth, box, hfov=2.0944):
    """box=(x,y,width,height), pixels in the original aligned RGB image.

    Return a visible surface point in camera_link (forward, left, up).
    Central-patch median reduces isolated bad depths; it is not segmentation.
    """
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    rows, cols = depth.shape
    left, right = max(0, x), min(cols, x+w)
    top, bottom = max(0, y), min(rows, y+h)
    if right-left < 2 or bottom-top < 2:
        return None
    u, v = (left+right)/2, (top+bottom)/2
    rw, rh = max(1, (right-left)*0.15), max(1, (bottom-top)*0.15)
    x0, x1 = max(0, int(u-rw)), min(cols, int(math.ceil(u+rw)))
    y0, y1 = max(0, int(v-rh)), min(rows, int(math.ceil(v+rh)))
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch >= 0.4) & (patch <= 10.0)
    if valid.sum() < max(3, math.ceil(patch.size * 0.3)):
        return None
    z = float(np.median(patch[valid]))
    # Reject patches spanning substantially different surfaces.
    if float(np.median(np.abs(patch[valid]-z))) > max(0.15, 0.05*z):
        return None
    vv, uu = np.nonzero(valid & (np.abs(patch-z) <= max(0.1, 0.03*z)))
    if len(uu) < 3:
        return None
    u, v = float(np.median(uu+x0)), float(np.median(vv+y0))
    focal = cols/(2*math.tan(hfov/2))
    # Gazebo sensor is at camera_link, not the offset URDF optical links.
    return np.array([z, -(u-cols/2)*z/focal, -(v-rows/2)*z/focal])


def transform_point(point, translation, quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('Invalid TF quaternion')
    q /= norm
    xyz, w = q[:3], q[3]
    p = np.asarray(point, dtype=float)
    return p + 2*np.cross(xyz, np.cross(xyz, p)+w*p) + translation


class ArtefactTracks:
    """Same-class nearest-neighbour association and running mean.

    At most one observation per track per image. Nearby same-class objects
    can still merge; tune radius using measured errors and object spacing.
    """
    def __init__(self, radius=0.75):
        self.radius = radius
        self.tracks = []
        self.next_id = 0

    def update(self, observations, stamp):
        # Remove unconfirmed one-off detections after five seconds.
        self.tracks = [t for t in self.tracks if t['count'] >= 3 or stamp-t['last'] <= 5]
        used = set()
        for label, point in observations:
            point = np.asarray(point, dtype=float)
            if label == 'negative' or not np.isfinite(point).all():
                continue
            candidates = [(np.linalg.norm(t['point']-point), t) for t in self.tracks
                          if t['label'] == label and t['id'] not in used]
            distance, track = min(candidates, key=lambda pair: pair[0]) if candidates else (math.inf, None)
            if distance <= self.radius:
                track['count'] += 1
                track['point'] += (point-track['point'])/track['count']
                track['last'] = stamp
            else:
                track = dict(id=self.next_id, label=label, point=point.copy(), count=1, last=stamp)
                self.next_id += 1
                self.tracks.append(track)
            used.add(track['id'])
        return [t for t in self.tracks if t['count'] >= 3]
