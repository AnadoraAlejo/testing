"""Perception 2: transparent colour-and-shape detector for simulated artefacts."""


import math
import cv2
import numpy as np

from cave_explorer.advanced1 import Detection, StopSignDetector, iou


# OBJECT TYPES THAT CAN BE IDENTIFIED
LABELS = ('green_alien', 'green_crystals', 'white_sphere',
          'ice_formation', 'blue_mushrooms', 'stop_sign')


# FIND OBJECT OUTLINES IN A COLOUR MASK
# IGNORE SHAPES THAT ARE TOO SMALL 
def contours_from_mask(mask, min_area=300.0, max_fraction=0.25):

    # IMAGE HEIGHT AND WIDTH
    height, width = mask.shape

    # REMOVE SMALL NOISE 
    filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # FILL SMALL GAPS IN DETECTED OBJECTS
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # FIND OUTER CONTOURS FROM CLEANED MASK
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # STORE VALID CONTOURS AND THEIR SHAPE INFORMATION
    output = []
    for contour in contours:

        # CALCULATE OBJECT AREA IN PIXELS
        area = float(cv2.contourArea(contour))

        # IGNORE OBJECTS OUTSIDE THE ALLOWED AREA RANGE
        if not (min_area <= area <= max_fraction * height * width):
            continue

        # GET BOUNDING BOX X, Y, WIDTH AND HEIGHT
        x, y, w, h = cv2.boundingRect(contour)

        # IGNORE TINY BOXES OR OBJECTS TOUCHING IMAGE EDGES
        if min(w, h) < 12 or x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue

        # CALCULATE THE LENGTH OF THE OBJECT OUTLINE
        perimeter = cv2.arcLength(contour, True)

        # CIRCULARITY CLOSE TO 1 MEANS THE SHAPE IS MORE CIRCULAR
        circle = 4.0 * math.pi * area / perimeter**2 if perimeter > 1e-6 else 0

        # CONVEX HULL IS THE OUTER SHAPE WITHOUT INWARD GAPS
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

        # SOLIDITY COMPARES ACTUAL AREA WITH THE CONVEX HULL AREA
        solidity = area / hull_area if hull_area > 1e-6 else 0

        # SAVE THE BOX, AREA, CIRCULARITY AND SOLIDITY
        output.append(((x, y, w, h), area, circle, solidity))

    # RETURN ALL OBJECT SHAPES
    return output


# REMOVE OVERLAPPING DETECTIONS OF THE SAME OBJECT TYPE
def suppress(detections, threshold=0.45):

    # STORE FINAL DETECTIONS
    result = []

    # CHECK EACH DETECTED OBJECT LABEL SEPARATELY
    for label in sorted({d.label for d in detections}):
        remaining = [d for d in detections if d.label == label]

        # Contour masks are already disjoint per channel; retain larger boxes first.
        # SORT BOXES FROM LARGEST TO SMALLEST
        remaining.sort(key=lambda d: d.box[2] * d.box[3], reverse=True)

        # KEEP A BOX AND REMOVE OTHER BOXES WITH TOO MUCH OVERLAP
        while remaining:
            best = remaining.pop(0)
            result.append(best)

            # IOU MEASURES HOW MUCH TWO BOUNDING BOXES OVERLAP
            remaining = [d for d in remaining if iou(best.box, d.box) < threshold]

    # RETURN DETECTIONS AFTER OVERLAP REMOVAL
    return result


def _stop_sign_red(hsv):
    # Stop-sign posters are red. The real artefacts are not.
    low = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([12, 255, 255]))
    high = cv2.inRange(hsv, np.array([168, 90, 70]), np.array([179, 255, 255]))
    return cv2.bitwise_or(low, high)


def _sign_zones(red):
    # The poster is bigger than the red sign, so nearby colour blobs belong to it too.
    height, width = red.shape
    zones = []
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 200:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        pad_x, pad_y = int(0.5 * w), int(0.6 * h)
        zones.append((max(0, x - pad_x), max(0, y - pad_y),
                      min(width, x + w + pad_x), min(height, y + h + pad_y)))
    return zones


def _on_stop_sign(red, zones, box):
    x, y, w, h = [int(v) for v in box]
    cx, cy = x + w / 2.0, y + h / 2.0
    if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in zones):
        return True
    height, width = red.shape
    pad = int(max(6, 0.2 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    region = red[y0:y1, x0:x1]
    if region.size == 0:
        return False
    red_pixels = int(cv2.countNonZero(region))
    return red_pixels >= 80 and red_pixels >= 0.02 * region.size


# IDENTIFY OBJECTS USING THEIR COLOUR AND SHAPE
class ColourShapeDetector:
    """Receives BGR OpenCV frames and returns original-resolution bounding boxes."""

    # USE ALL LABELS EXCEPT STOP SIGN (HANDLED BY ANOTHER DETECTOR)
    labels = LABELS[:-1]

    def __init__(self, min_area=300):
        # SET MINIMUM DETECTION AREA
        self.min_area = float(min_area)

    def detect(self, bgr):

        # CHECK THAT IMAGE HAS HEIGHT, WIDTH AND THREE COLOUR CHANNELS
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Expected BGR HxWx3 image')

        # CONVERT BGR IMAGE TO HSV FOR COLOUR DETECTION
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # STORE OBJECTS FOUND IN THIS IMAGE
        detections = []

        # FIND GREEN AREAS USING LOWER AND UPPER HSV LIMITS
        green = cv2.inRange(hsv, np.array([35, 65, 35]), np.array([95, 255, 255]))
        sign_red = _stop_sign_red(hsv)
        sign_zones = _sign_zones(sign_red)

        # CHECK GREEN SHAPES TO DISTINGUISH ALIENS FROM CRYSTALS
        for box, area, circularity, solidity in contours_from_mask(green, self.min_area):
            if _on_stop_sign(sign_red, sign_zones, box):
                continue

            # ASPECT RATIO = OBJECT HEIGHT DIVIDED BY WIDTH
            aspect = box[3] / max(1, box[2])

            # TALLER AND MORE SOLID GREEN SHAPES ARE CLASSIFIED AS ALIENS
            # OTHERWISE CLASSIFY AS GREEN CRYSTALS
            label = 'green_alien' if aspect >= 1.15 and solidity >= 0.4 else 'green_crystals'
            detections.append(Detection(label, box))

        # FIND BLUE AREAS USING HSV COLOUR LIMITS
        blue = cv2.inRange(hsv, np.array([85, 55, 45]), np.array([145, 255, 255]))

        # LABEL ACCEPTED BLUE SHAPES AS BLUE MUSHROOMS
        for box, area, circularity, solidity in contours_from_mask(blue, self.min_area):
            if _on_stop_sign(sign_red, sign_zones, box):
                continue
            detections.append(Detection('blue_mushrooms', box))

        # FIND BRIGHT LOW-SATURATION AREAS FOR WHITE AND ICE OBJECTS
        white = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([179, 75, 255]))

        # CHECK WHITE OBJECTS USING AREA, SHAPE AND ASPECT RATIO
        for box, area, circularity, solidity in contours_from_mask(white, self.min_area * 1.2, 0.22):
            aspect = box[3] / max(1, box[2])

            if _on_stop_sign(sign_red, sign_zones, box):
                continue

            # MORE CIRCULAR WHITE SHAPES ARE SPHERES
            if circularity >= 0.62 and 0.70 <= aspect <= 1.35:
                detections.append(Detection('white_sphere', box))

            # LARGER LESS-CIRCULAR WHITE SHAPES ARE ICE FORMATIONS
            elif area >= self.min_area * 4 and circularity < 0.58:
                detections.append(Detection('ice_formation', box))

        # REMOVE DUPLICATE OR OVERLAPPING BOXES BEFORE RETURNING
        return suppress(detections)


# COMBINE COLOUR-SHAPE DETECTIONS WITH THE EXISTING STOP SIGN DETECTOR
class CompositeDetector:
    """Supports both the existing XML cascade and colour-shape detections."""

    # INCLUDE ALL OBJECT TYPES, INCLUDING STOP SIGN
    labels = LABELS

    def __init__(self, filename, enable_color=True):

        # ENABLE OR DISABLE THE COLOUR AND SHAPE DETECTOR
        self.colour = ColourShapeDetector() if enable_color else None

        # LOAD STOP SIGN DETECTOR IF XML FILE IS PROVIDED
        self.stop = StopSignDetector(filename) if filename else None

        # AT LEAST ONE DETECTOR MUST BE AVAILABLE
        if self.stop is None and self.colour is None:
            raise ValueError('At least one detector must be enabled')

    def detect(self, bgr):

        # RUN COLOUR DETECTION IF ENABLED
        results = self.colour.detect(bgr) if self.colour else []

        # ADD STOP SIGN DETECTIONS IF ENABLED
        if self.stop is not None:
            signs = self.stop.detect(bgr)
            results = [d for d in results if not any(iou(d.box, sign.box) > 0.05 for sign in signs)]
            results.extend(signs)

        # REMOVE OVERLAPPING DETECTIONS AND RETURN RESULTS
        return suppress(results)
