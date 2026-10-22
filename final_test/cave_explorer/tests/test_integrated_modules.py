"""Offline tests for the added Advanced2 geometry and Perception2 detector."""
import math
import unittest
import numpy as np
from cave_explorer.advanced2a import GeometryAnalyser
from cave_explorer.perception2_detector import ColourShapeDetector
import cv2


class TestCaveGeometry(unittest.TestCase):
    def make_cave(self):
        grid = np.full((70, 120), 100, dtype=np.int16)
        grid[5:65, 5:46] = 0            # left large room
        grid[31:40, 46:76] = 0          # narrow corridor, 1.8m across
        grid[10:61, 76:112] = 0         # right large room
        return grid

    def test_widest_and_bottleneck(self):
        grid = self.make_cave()
        result = GeometryAnalyser().analyse(grid.ravel(), 120, 70, .2, (0., 0., 0.), (5., 7.))
        self.assertIsNotNone(result['widest'])
        self.assertGreater(result['widest'].clearance_m, 2)
        self.assertIsNotNone(result['bottleneck'])
        self.assertLess(result['bottleneck'].clearance_m, 1.1)
        self.assertGreater(result['bottleneck'].clearance_m, 0.7)

    def test_unexplored_unknown_is_not_open(self):
        grid = np.full((40, 40), -1, dtype=np.int16)
        grid[12:28, 12:28] = 0
        result = GeometryAnalyser().analyse(grid.ravel(), 40, 40, 0.2, (0, 0, 0), (4, 4))
        self.assertIsNotNone(result['widest'])
        self.assertLess(result['widest'].clearance_m, 2)

    def test_rotated_map_origin(self):
        x, y = GeometryAnalyser.transform(0, 0, 1.0, (10, 20, math.pi/2))
        self.assertAlmostEqual(x, 9.5)
        self.assertAlmostEqual(y, 20.5)

    def test_invalid_resolution(self):
        with self.assertRaises(ValueError):
            GeometryAnalyser().analyse([0]*100, 10, 10, 0., (0,0,0), (1,1))


class TestColorDetection(unittest.TestCase):
    def test_blue_artifact_box(self):
        img = np.zeros((250, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (80, 65), (140, 145), (255, 0, 0), -1)
        dets = ColourShapeDetector().detect(img)
        blue = [d for d in dets if d.label == 'blue_mushrooms']
        self.assertTrue(blue)
        x, y, w, h = blue[0].box
        self.assertLessEqual(x, 80)
        self.assertGreaterEqual(x+w, 140)

    def test_no_detection_on_black(self):
        img = np.zeros((250, 320, 3), dtype=np.uint8)
        self.assertEqual(ColourShapeDetector().detect(img), [])


if __name__ == '__main__':
    unittest.main()
