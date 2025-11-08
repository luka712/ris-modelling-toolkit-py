import unittest
import numpy as np

from ris_modelling_toolkit.src.data import UVEdgeBoundary, Axis2D


class DataUnitTests(unittest.TestCase):

    def test_create_uv_boundary_coord(self):

        # Just example of what makes sense for this class
        # If UVs cross from 0.8 to 1.2 on the U axis the crossed edge should be 1.0
        boundary = UVEdgeBoundary(crossed_edges=np.array([1.0]),
                                  axis=Axis2D.X,
                                  edge0=0.8, edge1=1.2,
                                  index0=0, index1=1,
                                  v0=np.array([0.0, 0.0, 0.0]), v1=np.array([1.0, 1.0, 1.0]),
                                  uv0=np.array([0.8, 0.5]), uv1=np.array([1.2, 0.5]))

        self.assertEqual(boundary.crossed_edges.tolist(), [1.0])
        self.assertEqual(boundary.uv_axis, Axis2D.X)
        self.assertEqual(boundary.min, 0.8)
        self.assertEqual(boundary.max, 1.2)
        self.assertEqual(boundary.index0, 0)
        self.assertEqual(boundary.index1, 1)
        np.testing.assert_array_equal(boundary.v0, np.array([0.0,
                                                            0.0,
                                                            0.0]))
        np.testing.assert_array_equal(boundary.v1, np.array([1.0,
                                                            1.0,
                                                            1.0]))
        np.testing.assert_array_equal(boundary.uv0, np.array([0.8,
                                                             0.5]))
        np.testing.assert_array_equal(boundary.uv1, np.array([1.2,
                                                             0.5]))


if __name__ == '__main__':
    unittest.main()