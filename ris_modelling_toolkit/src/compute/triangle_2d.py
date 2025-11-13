from typing import Tuple, Optional

from ris_modelling_toolkit.src.compute.constants import EPS_ABS_ZERO, EPS_REL_VALID


def twice_signed_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Compute twice the signed area of the triangle formed by three 2D points.

    The result equals the 2D cross product of the vectors (p2 - p1) and (p3 - p1):
        2A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    This value encodes both the magnitude and the orientation (winding order)
    of the triangle formed by the three points.

    Interpretation:
        - result > 0 → counterclockwise orientation
        - result < 0 → clockwise orientation
        - result == 0 → points are collinear

    To get the actual (unsigned) area:
        area = abs(twice_signed_area(p1, p2, p3)) / 2

    Returning "twice the signed area" avoids unnecessary division and is
    commonly used in computational geometry for orientation tests and
    polygon area calculations.

    :param p1: The first 2D point (x, y).
    :param p2: The second 2D point (x, y).
    :param p3: The third 2D point (x, y).
    :return: Twice the signed area of the triangle formed by the points.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

def barycentric_coordinates(
    point: Tuple[float, float],
    v0: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    """
    Compute the barycentric coordinates of a 2D point relative to a triangle.

    Barycentric coordinates (b0, b1, b2) express a point as a weighted combination
    of the triangle's vertices:

        point = b0 * v0 + b1 * v1 + b2 * v2

    The weights satisfy b0 + b1 + b2 = 1.
    This representation is widely used in interpolation (e.g., texture mapping, shading, or UV mapping).

    The computation is based on signed triangle areas:
        - The total area (a0) is twice the signed area of the full triangle (v0, v1, v2)
        - a1, a2, a3 are twice the signed sub-triangle areas formed with the target point
        - Each barycentric coordinate bi = ai / a0

    The result is validated to ensure:
        - The point lies inside (or very close to) the triangle
        - The barycentric weights sum to approximately 1
        - All weights are non-negative (within tolerance)

    :param point: The 2D point (x, y) to compute barycentric coordinates for.
    :param v0: The first vertex of the triangle (x, y).
    :param v1: The second vertex of the triangle (x, y).
    :param v2: The third vertex of the triangle (x, y).
    :return:
        A tuple of barycentric coordinates (b0, b1, b2) if the point lies within
        the triangle or on its edge; otherwise, None if the triangle is degenerate
        or the point lies outside within tolerance limits.

    Example:
        >>> barycentric_coordinates((0.25, 0.25), (0, 0), (1, 0), (0, 1))
        (0.5, 0.25, 0.25)

    Dependencies:
        Requires a helper function `twice_signed_area(p1, p2, p3)` that computes
        twice the signed area of a triangle.

    Notes:
        - Uses EPS_ABS_ZERO and EPS_REL_VALID for numerical stability.
        - The coordinates may slightly exceed [0, 1] due to floating-point precision.

    """
    a0 = twice_signed_area(v0, v1, v2)
    if abs(a0) < EPS_ABS_ZERO:
        # Degenerate triangle (zero area)
        return None

    # Compute signed sub-triangle areas relative to the target point
    a1 = twice_signed_area(point, v1, v2)
    a2 = twice_signed_area(v0, point, v2)
    a3 = twice_signed_area(v0, v1, point)

    # Normalize to get barycentric coordinates
    b0, b1, b2 = a1 / a0, a2 / a0, a3 / a0

    # Validate the result
    if abs(b0 + b1 + b2 - 1.0) > EPS_REL_VALID or min(b0, b1, b2) < -EPS_REL_VALID:
        return None

    return (b0, b1, b2)