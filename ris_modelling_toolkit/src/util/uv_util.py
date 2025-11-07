import math

import numpy
import numpy as np

from ris_modelling_toolkit.src.data import Axis2D, UVBoundaryCoordInfo, UVBoundaryPointInfo
from ris_modelling_toolkit.src.util.math_util import map_range, lerp


def find_uv_outside_bounds(
        index0: int,
        index1: int,
        v0: np.ndarray[list[float]],
        v1: np.ndarray[list[float]],
        uv0: np.ndarray[list[float]],
        uv1: np.ndarray[list[float]],
        axis: Axis2D) -> UVBoundaryCoordInfo | None:
    """
    Find integer boundaries outside [0,1] range between left and right UV coordinate.
    1. Determine the min and max of the two UV coordinates.
    2. If both are within [0,1], return None.
    3. Otherwise, calculate the integer boundaries outside [0,1] that lie between
    the min and max UVs.
    4. Return these boundaries as a numpy array.
    :param index0: The first vertex index of the face.
    :param index1: The second vertex index of the face.
    :param v0 : The first 3D vertex of the edge.
    :param v1 : The second 3D vertex of the edge.
    :param uv0 : The first vertex UV coordinates.
    :param uv1 : The second vertex UV coordinates.
    :param face0: The index of the first face vertex.
    :param face1: The index of the second face vertex.
    :param axis: The UV axis (X or Y) for which to check where boundary crossing occurs.
    :return: The list of integer boundaries outside [0,1] or None.
    """

    if len(v0) != 3 or len(v1) != 3:
        raise ValueError("v0 and v1 must be 3D coordinates.")

    if len(uv0) != 2 or len(uv1) != 2:
        raise ValueError("uv0 and uv1 must be 2D UV coordinates.")

    uv_coord_min = min(uv0[axis.value], uv1[axis.value])
    uv_coord_max = max(uv0[axis.value], uv1[axis.value])

    # If both are within the 0 to 1 range, return nothing.
    if uv_coord_min >= 0.0 and uv_coord_max <= 1.0:
        return None

    # Ceil to next integer as we don't cate about values below min if both are positive.
    # For example for min of 2.3 we want to start at 3.
    if uv_coord_min > 0.0:
        uv_coord_min = math.ceil(uv_coord_min)
    else:
        uv_coord_min = math.floor(uv_coord_min)

    if uv_coord_max > 0.0:
        uv_coord_max = math.floor(uv_coord_max)
    else:
        uv_coord_max = math.ceil(uv_coord_max)

    result = []
    f = uv_coord_min
    while f < uv_coord_max:
        if f >= 0.0 and f != uv_coord_min:
            result.append(math.floor(f))
        elif f < 0.0 and f != uv_coord_max:
            result.append(math.ceil(f))

        f += 1.0

    # If no edges were crossed, return None.
    if len(result) == 0.0:
        return None

    return UVBoundaryCoordInfo(crossed_edges=np.array(result),
                               index0=index0, index1=index1,
                               v0=v0, v1=v1,
                               uv0= uv0, uv1=uv1,
                               axis=axis,
                               edge0=uv_coord_min,
                               edge1=uv_coord_max)


def compute_uv_boundary_points(boundary_info: UVBoundaryCoordInfo | None) -> list[UVBoundaryPointInfo] | None:
    """
    Find 3D points on the edge between points a and b where UV coordinates cross specified boundaries.
    1. For each boundary in out_of_bound_points, map it to a parameter t in [0, 1] based on uv_a and uv_b.
    2. Use linear interpolation (lerp) to find the corresponding 3D point on the edge.
    3. Collect and return all such points.
    :param boundary_info: The class that holds list of UV boundaries to check.
    :return: The list of 3D points where UVs cross the boundaries or None.
    """

    if boundary_info is None:
        return None

    # Get coordinates for the specified axis. This will be either U or V.
    axis = boundary_info.uv_axis
    uv_coord_edge0 = boundary_info.edge0
    uv_coord_edge1 = boundary_info.edge1

    # Get coordinates for the other axis.
    other_axis = Axis2D.Y if axis == Axis2D.X else Axis2D.X
    other_uv_coord_edge0 = boundary_info.uv0[other_axis.value]
    other_uv_coord_edge1 = boundary_info.uv1[other_axis.value]

    boundaries = []
    # Check each crossed edge and compute the corresponding 3D point.
    # Crossed edges are integer values where UVs cross boundaries outside [0,1].
    for edge in boundary_info.crossed_edges:

        # Map the edge to a parameter t in [0, 1].
        t = map_range(edge, uv_coord_edge0, uv_coord_edge1, 0.0, 1.0)

        # Compute the 3D point using linear interpolation.
        point = lerp(boundary_info.v0, boundary_info.v1, t)

        # Compute the interpolated coordinate on the other UV axis.
        other_coord = lerp(other_uv_coord_edge0, other_uv_coord_edge1, t)

        # Create and store the boundary point info. This is information about the 3D point where boundary crossing occurs.
        boundaries.append(UVBoundaryPointInfo(
            point=point,
            index0=boundary_info.index0,
            index1=boundary_info.index1,
            v0=boundary_info.v0,
            v1=boundary_info.v1,
            crossed_edge=edge,
            edge0 = uv_coord_edge0,
            edge1 = uv_coord_edge1,
            uv_axis=axis,
            other_coord_interpolated=other_coord
        ))

    return boundaries