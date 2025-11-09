import unittest

import numpy as np

from ris_modelling_toolkit.src.data.plane import Plane
from ris_modelling_toolkit.src.data.triangle import Triangle


class TriangleTests(unittest.TestCase):

    def test_clip_triangle_by_plane(self):
        triangle = Triangle(
            v0=np.vstack([0, 0,0]),
            v1=np.vstack([2, 0,0]),
            v2=np.vstack([0, 1,0]),
            uv0=np.vstack([0, 0]),
            uv1=np.vstack([1, 0]),
            uv2=np.vstack([0, 1])
        )

        plane = Plane(point=np.array([1, 0, 0]), normal=np.array([0, 0, 1]))

        clipped_triangles = triangle.clip_by_plane(plane)
        self.assertEqual(len(clipped_triangles), 2)
        self.assertTrue(all(isinstance(t, Triangle) for t in clipped_triangles))




if __name__ == '__main__':
    unittest.main()