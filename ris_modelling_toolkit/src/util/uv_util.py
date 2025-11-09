import math
import numpy as np

from ris_modelling_toolkit.src.data import Axis2D, UVEdgeBoundary, UVBoundaryIntersection
from ris_modelling_toolkit.src.util.math_util import map_range, lerp


def find_uv_outside_bounds(
        v0: np.ndarray[list[float]],
        v1: np.ndarray[list[float]],
        uv0: np.ndarray[list[float]],
        uv1: np.ndarray[list[float]],
        axis: Axis2D) -> UVEdgeBoundary | None:
    """
       Detect whether the UV segment between two vertices crosses any integer UV tile boundaries.

       UV coordinates outside the normalized [0, 1] range imply tiled textures. If the UV
       values at the ends of an edge lie on opposite sides of an integer boundary (e.g.,
       0.8 → 1.2 crosses U = 1), the mesh must later be split at that boundary to prevent
       incorrect texture interpolation.

       This function:
         1. Extracts the UV coordinate on the specified axis (U or V).
         2. Checks whether the segment spans any integer boundaries.
         3. If so, returns a `UVEdgeBoundary` describing the boundaries crossed and the
            geometric and UV data needed for interpolation later.
         4. If not, returns None.

       Parameters
       ----------
       v0, v1 : np.ndarray, shape (3,)
           The 3D positions of the edge endpoints.
       uv0, uv1 : np.ndarray, shape (2,)
           The UV coordinates of the edge endpoints.
       axis : Axis2D
           Which UV axis (U or V) to analyze.

       Returns
       -------
       UVEdgeBoundary or None
           A boundary descriptor if integer UV boundaries are crossed, otherwise None.

       Notes
       -----
       - Only boundaries outside the [0, 1] range are considered.
       - This function does not compute 3D split points; it only identifies where splits
         *should* occur. The actual intersection vertex is computed later.
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

    return UVEdgeBoundary(crossed_edges=np.array(result),
                               v0=v0, v1=v1,
                               uv0=uv0, uv1=uv1,
                               axis=axis,
                               edge0=uv_coord_min,
                               edge1=uv_coord_max)


def compute_uv_boundary_points(boundary_info: UVEdgeBoundary | None) -> list[UVBoundaryIntersection] | None:
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
        boundaries.append(UVBoundaryIntersection(
            point=point,
            v0=boundary_info.v0,
            v1=boundary_info.v1,
            crossed_edge=edge,
            edge0 = uv_coord_edge0,
            edge1 = uv_coord_edge1,
            uv_axis=axis,
            other_coord_interpolated=other_coord
        ))

    return boundaries