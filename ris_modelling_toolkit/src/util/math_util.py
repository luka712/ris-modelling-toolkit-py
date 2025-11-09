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


def lerp(start: numpy.ndarray[list[float]] | float,
         end: numpy.ndarray[list[float]] | float,
         t: float) -> numpy.ndarray[list[float]] | float:
    """
    Linearly interpolates between two points.
    :param start: The starting point.
    :param end: The ending point.
    :param t: The interpolation factor (0.0 to 1.0).
    :return: The interpolated point.
    """
    return (1 - t) * start + t * end


def is_valid_triangle(v0: np.ndarray[list[float]],
                      v1: np.ndarray[list[float]],
                      v2: np.ndarray[list[float]],
                      tol: float = 1e-8) -> bool:
    """
    Check if three 3D points form a valid triangle (not degenerate).

    :param v0: np.ndarray of shape (3,), first vertex of the triangle.
    :param v1: np.ndarray of shape (3,), second vertex of the triangle.
    :param v2: np.ndarray of shape (3,), third vertex of the triangle.
    :param tol: Tolerance for numerical precision.
    :return: True if the points form a valid triangle, False otherwise.
    """

    if len(v0) != 3 or len(v1) != 3 or len(v2) != 3:
        raise ValueError("v0, v1, and v2 must be 3D coordinates.")

    # Compute the area of the triangle using the cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_product = np.cross(edge1, edge2)
    area = np.linalg.norm(cross_product) / 2.0

    return area > tol


def perpendicular_vector(v: np.ndarray[list[float]] | np.ndarray[tuple[float]]) -> np.ndarray[list[float]]:
    """
    Compute a vector that is perpendicular to the given 3D vector.
    :param v: The input 3D vector.
    :return: A 3D vector that is perpendicular to v.
    """
    if len(v) != 3:
        raise ValueError("Input vector must be a 3D coordinate.")

    if np.allclose(v, np.zeros(3)):
        raise ValueError("Cannot compute a perpendicular vector to the zero vector.")

    # Find a vector that is not parallel to v
    if not np.isclose(v[0], 0) or not np.isclose(v[1], 0):
        arbitrary = np.array([-v[1], v[0], 0])
    else:
        arbitrary = np.array([0, -v[2], v[1]])

    # Compute the cross product to get a perpendicular vector
    perp_vector = np.cross(v, arbitrary)
    perp_vector_normalized = perp_vector / np.linalg.norm(perp_vector)

    return perp_vector_normalized
