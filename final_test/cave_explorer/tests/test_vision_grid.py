import unittest
import math
import numpy as np
from cave_explorer.advanced1 import Detection,degrade,restore,match_counts
from cave_explorer.planning_support import GridPlanner
from cave_explorer.perception3_geometry import depth_point,transform_point


class VisionTests(unittest.TestCase):
    def test_noise_reproducible_and_no_source_mutation(self):
        original=np.full((80,100,3),128,np.uint8)
        first=degrade(original,'noise',3,42)
        np.testing.assert_array_equal(first,degrade(original,'noise',3,42))
        self.assertGreater(first.std(),degrade(original,'noise',1,42).std())
        self.assertTrue((original==128).all())
    def test_light_severity(self):
        original=np.full((80,100,3),200,np.uint8)
        values=[degrade(original,'low_light',i).mean() for i in (0,1,2,3)]
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))
    def test_restoration_contract(self):
        source=np.random.default_rng(1).integers(0,50,(40,60,3),dtype=np.uint8)
        restored=restore(source)
        self.assertEqual(restored.shape,source.shape)
        self.assertEqual(restored.dtype,np.uint8)
    def test_one_to_one_false_positive_and_negative(self):
        box=Detection('stop_sign',(0,0,20,20))
        other=Detection('green_alien',(0,0,20,20))
        self.assertEqual(match_counts([box,box],[box]),(1,1,0))
        self.assertEqual(match_counts([other],[box]),(0,1,1))
        self.assertEqual(match_counts([box],[]),(0,1,0))
        self.assertEqual(match_counts([],[]),(0,0,0))
    def test_depth_and_transform_axes(self):
        depth=np.full((480,720),4.)
        p=depth_point(depth,(350,230,20,20))
        np.testing.assert_allclose(p,[4,0,0],atol=.02)
        self.assertLess(depth_point(depth,(450,230,20,20))[1],0)
        rotated=transform_point([4,0,0],np.array([1,2,3]),[0,0,2**-.5,2**-.5])
        np.testing.assert_allclose(rotated,[1,6,3],atol=1e-9)
        self.assertIsNone(depth_point(np.zeros_like(depth),(350,230,20,20)))


class GridTests(unittest.TestCase):
    def test_unknown_and_wall_block_candidates(self):
        grid=GridPlanner()
        cells=np.zeros((60,60),np.int16); cells[:,30]=100; cells[0:8,:]=-1
        grid.update(cells,60,60,.2,(0,0,0))
        self.assertTrue(grid.safe_goal((3,4),(2,4)))
        self.assertFalse(grid.safe_goal((9,4),(2,4)))
        self.assertFalse(grid.safe_goal((3,.4),(2,4)))
        goal=grid.explore((2,4))
        self.assertIsNotNone(goal)
        self.assertTrue(grid.safe_goal(goal,(2,4)))
    def test_rotated_map(self):
        grid=GridPlanner(); cells=np.zeros((40,40),np.int16)
        grid.update(cells,40,40,.2,(10,20,math.pi/2))
        xy=grid.world(15,20)
        self.assertEqual(grid.cell(xy),(15,20))
        self.assertTrue(grid.safe_goal(xy,xy))
    def test_approach_standoff_and_orientation(self):
        grid=GridPlanner(); grid.update(np.zeros((80,80),np.int16),80,80,.2,(0,0,0))
        goal=grid.approach((8,8,1),(4,8,0),standoff=2)
        np.testing.assert_allclose(goal,(6,8,0),atol=1e-9)
