import unittest
import numpy as np

from ris_modelling_toolkit.src.data import UVBoundaryCoordInfo, Axis2D


class DataUnitTests(unittest.TestCase):

    def test_create_uv_boundary_coord(self):

        # Just example of what makes sense for this class
        # If UVs cross from 0.8 to 1.2 on the U axis the crossed edge should be 1.0
        boundary = UVBoundaryCoordInfo(crossed_edges=np.array([1.0]), axis=Axis2D.X, edge0=0.8, edge1=1.2)

        self.assertEqual(boundary.crossed_edges.tolist(), [1.0])
        self.assertEqual(boundary.uv_axis, Axis2D.X)
        self.assertEqual(boundary.min, 0.8)
        self.assertEqual(boundary.max, 1.2)


if __name__ == '__main__':
    unittest.main()