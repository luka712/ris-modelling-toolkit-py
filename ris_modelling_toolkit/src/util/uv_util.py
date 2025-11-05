import math

import numpy
import numpy as np

from ris_modelling_toolkit.src.data import Axis2D, UVBoundaryCoordInfo, UVBoundaryPointInfo
from ris_modelling_toolkit.src.util.math_util import map_range, lerp


def find_uv_outside_bounds(left: float, right: float, axis: Axis2D) -> UVBoundaryCoordInfo | None:
    """
    Find integer boundaries outside [0,1] range between left and right UV coordinate.
    1. Determine the min and max of the two UV coordinates.
    2. If both are within [0,1], return None.
    3. Otherwise, calculate the integer boundaries outside [0,1] that lie between
    the min and max UVs.
    4. Return these boundaries as a numpy array.
    :param left: The left UV coordinate.
    :param right: The right UV coordinate.
    :param axis: The UV axis (X or Y) where the boundary crossing occurs.
    :return: The list of integer boundaries outside [0,1] or None.

    # Example:
    >>> result = find_uv_outside_bounds(-1.5, 3.3, Axis2D.X)
    >>> print(result.crossed_edges.tolist())
    [-1, 0, 1, 2, 3]
    >>> print(result.uv_axis)
    Axis2D.X
    >>> print(result.min)
    -1.5
    >>> print(result.max)
    3.3
    """

    uv_min = min(left, right)
    uv_max = max(left, right)

    # If both are within the 0 to 1 range, return nothing.
    if uv_min >= 0.0 and uv_max <= 1.0:
        return None

    # Ceil to next integer as we don't cate about values below min if both are positive.
    # For example for min of 2.3 we want to start at 3.
    if uv_min > 0.0 and uv_max > 0.0:
        uv_min = math.ceil(uv_min)

    # 0 for min is OK therefore we can start at 1.
    if uv_min == 0.0:
        uv_min = 1.0

     # Floor or Ceil max based on if it's negative or positive.
    if uv_max < 0.0:
        uv_max = math.floor(uv_max)
    else:
        uv_max = math.ceil(uv_max)

    result = []
    f = uv_min
    while f < uv_max:
        if f >= 0.0:
            result.append(math.floor(f))
        else:
            if f < -1.0: # to avoid adding -0 as it's edge
                result.append(math.ceil(f))

        f += 1.0

    # If no edges were crossed, return None.
    if len(result) == 0.0:
        return None

    return UVBoundaryCoordInfo(crossed_edges=np.array(result), axis=axis, edge0=left, edge1=right)


def compute_uv_boundary_points(a: numpy.ndarray[tuple[float]],
                               b: numpy.ndarray[tuple[float]],
                               uv_a: list[float],
                               uv_b: list[float],
                               axis: Axis2D,
                               boundary_info: UVBoundaryCoordInfo) -> list[UVBoundaryPointInfo] | None:
    """
    Find 3D points on the edge between points a and b where UV coordinates cross specified boundaries.
    1. For each boundary in out_of_bound_points, map it to a parameter t in [0, 1] based on uv_a and uv_b.
    2. Use linear interpolation (lerp) to find the corresponding 3D point on the edge.
    3. Collect and return all such points.
    :param a: The starting 3D point.
    :param b: The ending 3D point.
    :param uv_a: The UV coordinate at point a.
    :param uv_b: The UV coordinate at point b.
    :param axis: The axis (X or Y) to consider for UV coordinates.
    :param boundary_info: The class that holds list of UV boundaries to check.
    :return: The list of 3D points where UVs cross the boundaries or None.
    """

    if boundary_info is None:
        return None

    if boundary_info.uv_axis != axis:
        raise ValueError("Mismatched UV axis in boundary info.")

    axis_index = axis.value
    uv_a = uv_a[axis_index]
    uv_b = uv_b[axis_index]

    points = []
    for edge in boundary_info.crossed_edges:

        t = map_range(edge, uv_a, uv_b, 0.0, 1.0)
        p = lerp(a, b, t)
        points.append(UVBoundaryPointInfo(point=p, uv_axis=axis))

    return points