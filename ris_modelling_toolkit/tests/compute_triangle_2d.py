import unittest
from math import isclose

from ris_modelling_toolkit.src.compute.triangle_2d import twice_signed_area, barycentric_coordinates


class ComputeTriangle2D(unittest.TestCase):

    def test_counterclockwise_triangle(self):
        p1, p2, p3 = (0, 0), (1, 0), (0, 1)
        result = twice_signed_area(p1, p2, p3)
        self.assertTrue(isclose(result, 1.0))
        self.assertGreater(result, 0)  # CCW orientation

    def test_clockwise_triangle(self):
        p1, p2, p3 = (0, 0), (0, 1), (1, 0)
        result = twice_signed_area(p1, p2, p3)
        self.assertTrue(isclose(result, -1.0))
        self.assertLess(result, 0)  # CW orientation

    def test_collinear_points(self):
        p1, p2, p3 = (0, 0), (1, 1), (2, 2)
        result = twice_signed_area(p1, p2, p3)
        self.assertTrue(isclose(result, 0.0))

    def test_negative_coordinates(self):
        p1, p2, p3 = (-1, -1), (0, -1), (-1, 0)
        result = twice_signed_area(p1, p2, p3)
        self.assertTrue(isclose(result, 1.0))
        self.assertGreater(result, 0)

    def test_area_magnitude_symmetry(self):
        p1, p2, p3 = (0, 0), (2, 0), (0, 2)
        ccw = twice_signed_area(p1, p2, p3)
        cw  = twice_signed_area(p1, p3, p2)
        self.assertTrue(isclose(ccw, -cw))
        self.assertTrue(isclose(abs(ccw), 4.0))  # Twice area = base*height = 4

    def test_small_floating_point(self):
        p1, p2, p3 = (0.0, 0.0), (1e-9, 0.0), (0.0, 1e-9)
        result = twice_signed_area(p1, p2, p3)
        self.assertTrue(isclose(result, 1e-18))

    def assertTupleAlmostEqual(self, t1, t2, tol=1e-9):
        """Helper to compare tuples of floats approximately."""
        self.assertTrue(all(isclose(a, b, abs_tol=tol) for a, b in zip(t1, t2)),
                        f"{t1} != {t2}")

    def test_point_at_vertex(self):
        """A vertex should have barycentric coordinate (1, 0, 0) or equivalent."""
        v0, v1, v2 = (0, 0), (1, 0), (0, 1)
        b = barycentric_coordinates(v0, v0, v1, v2)
        self.assertTupleAlmostEqual(b, (1.0, 0.0, 0.0))

    def test_point_at_center(self):
        """The centroid should have equal barycentric weights (1/3, 1/3, 1/3)."""
        v0, v1, v2 = (0, 0), (1, 0), (0, 1)
        centroid = ((v0[0] + v1[0] + v2[0]) / 3, (v0[1] + v1[1] + v2[1]) / 3)
        b = barycentric_coordinates(centroid, v0, v1, v2)
        self.assertTupleAlmostEqual(b, (1 / 3, 1 / 3, 1 / 3), tol=1e-6)

    def test_point_on_edge(self):
        """A point on an edge should have one weight = 0."""
        v0, v1, v2 = (0, 0), (1, 0), (0, 1)
        midpoint = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2)
        b = barycentric_coordinates(midpoint, v0, v1, v2)
        # Midpoint on v0-v1 should give roughly (0.5, 0.5, 0)
        self.assertTupleAlmostEqual(b, (0.5, 0.5, 0.0), tol=1e-6)

    def test_point_outside_triangle(self):
        """A point clearly outside the triangle should return None."""
        v0, v1, v2 = (0, 0), (1, 0), (0, 1)
        p_out = (1.0, 1.0)
        b = barycentric_coordinates(p_out, v0, v1, v2)
        self.assertIsNone(b)

    def test_degenerate_triangle(self):
        """Collinear triangle should return None (zero area)."""
        v0, v1, v2 = (0, 0), (1, 1), (2, 2)
        b = barycentric_coordinates((1, 1), v0, v1, v2)
        self.assertIsNone(b)

    def test_negative_coordinates(self):
        """Barycentric coordinates should still be valid for triangles in any quadrant."""
        v0, v1, v2 = (-1, -1), (0, -1), (-1, 0)
        p = (-0.75, -0.75)  # near center
        b = barycentric_coordinates(p, v0, v1, v2)
        self.assertIsNotNone(b)
        self.assertTrue(all(bi >= -1e-6 for bi in b))  # non-negative within tolerance
        self.assertTrue(isclose(sum(b), 1.0, abs_tol=1e-6))

    def test_point_near_edge_precision(self):
        """Check that a point slightly off the edge within EPS tolerance still passes."""
        v0, v1, v2 = (0, 0), (1, 0), (0, 1)
        p = (0.5, -1e-7)  # just below the base edge
        b = barycentric_coordinates(p, v0, v1, v2)
        self.assertIsNotNone(b)

if __name__ == "__main__":
    unittest.main()
