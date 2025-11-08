import unittest
import numpy as np

from ris_modelling_toolkit.src.data import UVEdgeBoundary, Axis2D
from ris_modelling_toolkit.src.util import compute_uv_boundary_points, find_uv_outside_bounds


class UVUtilUnitTests(unittest.TestCase):

    def test_find_uv_outside_bounds(self):

        result = find_uv_outside_bounds(
            uv0=np.array([-1.5, 0.0]),
            uv1=np.array([2.3, 0.0]),
            v0=np.array([0.0, 0.0, 0.0]),
            v1=np.array([5.0, 0.0, 0.0]),
            index0=0,
            index1=1,
            axis=Axis2D.X
        )
        self.assertIsNotNone(result)

        self.assertEqual(result.crossed_edges.tolist(), [-1, 0, 1, 2])
        self.assertEqual(result.uv_axis, Axis2D.X)
        self.assertEqual(result.min, -1.5)
        self.assertEqual(result.max, 2.3)


    def test_compute_uv_boundary_points(self):
        coords = UVEdgeBoundary(
            crossed_edges=np.array([-1,0,1,2,3]),
            axis=Axis2D.X, edge0=-1.0, edge1=4.0,
            v0=np.array([0,0,0]), v1=np.array([1,0,0]),
            index0=0, index1=1,
            uv0=np.array([-1.0, 0.0]), uv1=np.array([4.0, 0.0])
        )

        a = np.array([0.0, 0.0, 0.0])
        b = np.array([5.0, 0.0, 0.0])
        result = compute_uv_boundary_points(a, b, [-1.0, 0.0], [4.0, 0.0], Axis2D.X, coords)

        print(result)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        expected_points = [np.array([0.0, 0.0, 0.0]),
                           np.array([1.0, 0.0, 0.0]),
                           np.array([2.0, 0.0, 0.0]),
                           np.array([3.0, 0.0, 0.0]),
                           np.array([4.0, 0.0, 0.0])]

        for i in range(len(expected_points)):
            self.assertEqual(result[i].point.tolist(), expected_points[i].tolist())


if __name__ == '__main__':
    unittest.main()