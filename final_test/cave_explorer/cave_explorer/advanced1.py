"""Advanced 1: controlled degradations and a candidate robust pipeline.

All boxes use (x, y, width, height) in original image coordinates.
No clean-image pixels or ground truth are used by restore().
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    box: tuple


class StopSignDetector:
  
    labels = ('stop_sign',)

    def __init__(self, filename):
        if not hasattr(cv2, 'CascadeClassifier'):
            raise ValueError('This supplied Haar model needs OpenCV 4.x (Ubuntu python3-opencv)')
        self.model = cv2.CascadeClassifier(str(filename))
        if self.model.empty():
            raise ValueError('Cannot load stop-sign XML: '+str(filename))

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                           minSize=(20, 20))
        return [Detection('stop_sign', tuple(map(int, box))) for box in boxes]


def degrade(bgr, kind, level, seed=42):
    if kind not in ('none', 'low_light', 'noise') or level not in (0, 1, 2, 3):
        raise ValueError('kind: none/low_light/noise; level: 0..3')
    if kind == 'none' or level == 0:
        return bgr.copy()
    if kind == 'low_light':
        return np.clip(bgr.astype(np.float32)*[1, 0.65, 0.40, 0.20][level], 0, 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, [0, 10, 25, 45][level], bgr.shape)
    return np.clip(bgr.astype(np.float32)+noise, 0, 255).astype(np.uint8)


def restore(bgr):
   
    filtered = cv2.bilateralFilter(bgr, 5, 45, 45)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    median = float(np.median(l))/255.0
    if 0.005 < median < 0.40:
        gamma = float(np.clip(np.log(0.45)/np.log(median), 0.35, 1.0))
        table = np.round(255*(np.arange(256)/255.0)**gamma).astype(np.uint8)
        l = cv2.LUT(l, table)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    l = cv2.addWeighted(l, 0.5, enhanced, 0.5, 0)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def annotate(bgr, detections, title=''):
    result = bgr.copy()
    for d in detections:
        x, y, w, h = map(int, d.box)
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(result, d.label, (x,max(18,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if title:
        cv2.rectangle(result, (0,0), (result.shape[1],28), (0,0,0), -1)
        cv2.putText(result, title, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return result


def iou(a, b):
    ax,ay,aw,ah = a
    bx,by,bw,bh = b
    intersection = max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    union = aw*ah+bw*bh-intersection
    return intersection/union if union > 0 else 0.0


def match_counts(predictions, truth, threshold=0.5):
    graph = [[j for j,t in enumerate(truth) if p.label == t.label and iou(p.box,t.box)>=threshold]
             for p in predictions]
    assigned = {}
    def augment(i, seen):
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    tp = sum(augment(i,set()) for i in range(len(predictions)))
    return tp, len(predictions)-tp, len(truth)-tp
