
"""A collection of mathematical utility functions."""
import numpy
import numpy as np

def map_range(value: float, left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    """
    Maps a value from one range to another.
    :param value: The value to map.
    :param left_min: The minimum of the left range.
    :param left_max: The maximum of the left range.
    :param right_min: The minimum of the right range.
    :param right_max: The maximum of the right range.
    :return: The mapped value in the right range.
    """

    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = (value - left_min) / left_span

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


def lerp(start: numpy.ndarray[tuple[float]], end: numpy.ndarray[tuple[float]], t: float) -> numpy.ndarray:
    """
    Linearly interpolates between two points.
    :param start: The starting point.
    :param end: The ending point.
    :param t: The interpolation factor (0.0 to 1.0).
    :return: The interpolated point.
    """
    return (1 - t) * start + t * end


def point_in_triangle(point: np.ndarray[tuple[float]],
                      v0: np.ndarray[float],
                      v1: np.ndarray[float],
                      v2: np.ndarray[float],
                      tol: float = 1e-8) -> bool:
    """
    Check if a 3D point lies inside a triangle.

    :param point: np.ndarray of shape (3,), the 3D point to test.
    :param v0: np.ndarray of shape (3,), first vertex of the triangle.
    :param v1: np.ndarray of shape (3,), second vertex of the triangle.
    :param v2: np.ndarray of shape (3,), third vertex of the triangle.
    :param tol: Tolerance for numerical precision.
    :return: True if the point is inside the triangle (or on its edge), False otherwise.
    """
    # Vectors from v0
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0

    # Compute dot products
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    # Compute barycentric coordinates
    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < tol:
        return False  # Degenerate triangle

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    # Point is inside triangle if all barycentric coords are between 0 and 1
    return (-tol <= u <= 1 + tol) and (-tol <= v <= 1 + tol) and (-tol <= w <= 1 + tol)