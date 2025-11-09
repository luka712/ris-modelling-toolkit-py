import unittest

import numpy as np

from ris_modelling_toolkit.src.data.line import LineSegment


class LineSegmentTest(unittest.TestCase):

    def test_perpendicular_line(self):
        line_segment = LineSegment(
            start_point=np.array([0.0, 0.0, 0.0]),
            end_point=np.array([10.0, 0.0, 0.0]))

        point = np.array([5.0, 0.0, 0.0])

        perp_line = line_segment.perpendicular_line(point, length=10.0)
        expected_start = np.array([5.0, -5.0, 0.0])
        expected_end = np.array([5.0, 5.0])
        np.testing.assert_array_almost_equal(perp_line.start_point, expected_start)
        np.testing.assert_array_almost_equal(perp_line.end_point, expected_end)

if __name__ == '__main__':
    unittest.main()