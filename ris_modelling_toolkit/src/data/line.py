import numpy as np


class LineSegment:
    """The LineSegment class represents a line segment in 3D space defined by two endpoints."""

    def __init__(self,
                 start_point:  np.ndarray[list[float]],
                 end_point:  np.ndarray[list[float]]):
        """
        Initialize the LineSegment with start and end points.
        :param start_point: A numpy array representing the starting point of the line segment.
        :param end_point: A numpy array representing the ending point of the line segment.
        """

        if len(start_point) != 3 or len(end_point) != 3:
            raise ValueError("Start and end points must be 3D coordinates.")

        self.start_point = start_point
        self.end_point = end_point

    def is_point_on_segment(
            self,
            p: np.ndarray[list[float]],
            tol: float = 1e-8) -> bool:
        """
        Check if point p lies on the line segment defined by start_point and end_point.
        :param p: The point to test.
        :param tol: Tolerance for numerical precision.
        :return: True if point p lies on the line segment, False otherwise.
        """

        if len(p) != 3:
            raise ValueError("Points p must be 3D coordinates.")

        ab = self.end_point - self.start_point
        ap = p - self.start_point
        bp = p - self.end_point

        cross = np.cross(ab, ap)
        if np.linalg.norm(cross) > tol:
            return False  # Not collinear

        dot1 = np.dot(ab, ap)
        dot2 = np.dot(-ab, bp)

        return dot1 >= -tol and dot2 >= -tol

    def perpendicular_line(self,
                           p: np.ndarray[list[float]],
                           length: float = 10000) -> "LineSegment":
        if len(p) != 3:
            raise ValueError("Point p must be a 3D coordinate.")

        a = self.start_point
        b = self.end_point

        ab = b - a
        ap = p - a
        ab_norm = ab / np.linalg.norm(ab)
        projection_length = np.dot(ap, ab_norm)
        projection = ab_norm * projection_length
        foot_point = a + projection
        direction = p - foot_point
        direction_norm = direction / np.linalg.norm(direction)
        offset = direction_norm * (length / 2)
        perp_start = p - offset
        perp_end = p + offset
        return LineSegment(perp_start, perp_end)