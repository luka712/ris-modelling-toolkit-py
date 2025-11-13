import math
from typing import List, Tuple, Optional

from ris_modelling_toolkit.src.compute.constants import EPS_REL_VALID, EPS_ABS_ZERO
from ris_modelling_toolkit.src.compute.triangle_2d import barycentric_coordinates


def clean_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Cleans a polygon by removing redundant vertices and ensuring proper closure.

    This function performs the following operations:
      1. Removes consecutive duplicate points (within a small epsilon tolerance)
         to prevent degenerate polygon edges.
      2. Ensures the polygon is properly closed — if the first and last points
         are nearly identical, the redundant closing point is removed.
      3. Returns an empty list if the resulting polygon has fewer than three vertices,
         since it can no longer define a valid area.

    The cleanup process is useful before geometric operations such as triangulation,
    clipping, or collision detection, where precision and vertex uniqueness matter.

    :param poly: A list of (x, y) coordinate tuples representing the polygon vertices.
    :return: A cleaned list of (x, y) tuples with redundant points removed.
             Returns an empty list if the polygon is invalid (fewer than 3 vertices).
    """
    if len(poly) < 3:
        return []

    cleaned: List[Tuple[float, float]] = []
    prev_p: Optional[Tuple[float, float]] = None

    for p in poly:
        if prev_p is None or (abs(p[0] - prev_p[0]) > EPS_REL_VALID or abs(p[1] - prev_p[1]) > EPS_REL_VALID):
            cleaned.append(p)
            prev_p = p

    # Remove redundant closure point if first and last vertices are effectively identical
    if (
        len(cleaned) > 2
        and abs(cleaned[0][0] - cleaned[-1][0]) < EPS_REL_VALID
        and abs(cleaned[0][1] - cleaned[-1][1]) < EPS_REL_VALID
    ):
        cleaned.pop()

    return cleaned if len(cleaned) >= 3 else []


def _intersect_vertical(
    s: Tuple[float, float],
    e: Tuple[float, float],
    u_val: float
) -> Optional[Tuple[float, float]]:
    """
    Computes the intersection point between a line segment and a vertical line u = u_val.

    The function checks whether the line segment defined by points `s` and `e`
    crosses the vertical line at coordinate `u = u_val`. If an intersection occurs
    within the segment bounds (including a small epsilon tolerance), it returns
    the intersection point.

    This is typically used during polygon clipping (e.g., Sutherland–Hodgman)
    when trimming polygons against vertical boundaries.

    :param s: The start point of the line segment, as a tuple (u, v).
    :param e: The end point of the line segment, as a tuple (u, v).
    :param u_val: The fixed u-coordinate of the vertical line.
    :return: The intersection point (u_val, v_inter) if the segment crosses the
             vertical line within tolerance, otherwise None.
    """
    du = e[0] - s[0]
    if abs(du) < EPS_ABS_ZERO:
        return None  # Segment is (nearly) vertical; no unique intersection

    t = (u_val - s[0]) / du
    if -EPS_ABS_ZERO <= t <= 1 + EPS_ABS_ZERO:
        return (u_val, s[1] + t * (e[1] - s[1]))

    return None

def _intersect_horizontal(
    s: Tuple[float, float],
    e: Tuple[float, float],
    v_val: float
) -> Optional[Tuple[float, float]]:
    """
    Computes the intersection point between a line segment and a horizontal line v = v_val.

    The function determines whether the line segment connecting points `s` and `e`
    crosses the horizontal line at coordinate `v = v_val`. If an intersection occurs
    within the segment bounds (considering a small epsilon tolerance), it returns
    the corresponding intersection point.

    This function is typically used during polygon clipping operations
    (e.g., Sutherland–Hodgman) to trim polygons against horizontal boundaries.

    :param s: The start point of the line segment, as a tuple (u, v).
    :param e: The end point of the line segment, as a tuple (u, v).
    :param v_val: The fixed v-coordinate of the horizontal line.
    :return: The intersection point (u_inter, v_val) if the segment crosses the line
             within tolerance, otherwise None.
    """
    dv = e[1] - s[1]
    if abs(dv) < EPS_ABS_ZERO:
        return None  # Segment is (nearly) horizontal; no unique intersection

    t = (v_val - s[1]) / dv
    if -EPS_ABS_ZERO <= t <= 1 + EPS_ABS_ZERO:
        return (s[0] + t * (e[0] - s[0]), v_val)

    return None

def _clip_against_halfplane(
    poly: List[Tuple[float, float]],
    inside_func: callable,
    intersect_func: callable
) -> List[Tuple[float, float]]:
    """
    Clips a polygon against a single half-plane using the Sutherland–Hodgman algorithm.

    This function iterates over all edges of the input polygon and determines
    which vertices lie inside or outside the clipping half-plane. Based on
    the edge transitions, it adds vertices and intersection points to the
    output polygon.

    The clipping behavior is defined by two user-provided functions:
      - `inside_func(point)`: Returns True if the point lies inside (or on) the half-plane.
      - `intersect_func(s, e)`: Returns the intersection point between the edge (s → e)
        and the clipping boundary, or None if no valid intersection exists.

    This function does not assume convexity of the polygon and can be used
    successively to clip against multiple half-planes (e.g., rectangular regions).

    :param poly: A list of (x, y) points defining the polygon vertices in order.
    :param inside_func: Callable that returns True if a given point is inside the half-plane.
    :param intersect_func: Callable that computes the intersection between a polygon edge
                           and the clipping boundary when one vertex is inside and the other outside.
    :return: A list of (x, y) vertices representing the polygon after clipping.
             Returns an empty list if the polygon lies completely outside the half-plane.
    """
    if not poly:
        return []

    output: List[Tuple[float, float]] = []
    prev_point: Tuple[float, float] = poly[-1]
    prev_inside: bool = inside_func(prev_point)

    for curr_point in poly:
        curr_inside: bool = inside_func(curr_point)

        if curr_inside:
            if not prev_inside:
                # Edge enters the half-plane → add intersection point
                inter = intersect_func(prev_point, curr_point)
                if inter is not None:
                    output.append(inter)
            # Add current vertex if inside
            output.append(curr_point)
        elif prev_inside:
            # Edge exits the half-plane → add intersection point
            inter = intersect_func(prev_point, curr_point)
            if inter is not None:
                output.append(inter)

        prev_point = curr_point
        prev_inside = curr_inside

    return output



def clip_to_unit_square(
    poly: List[Tuple[float, float]],
    left: float,
    right: float,
    bottom: float,
    top: float
) -> List[Tuple[float, float]]:
    """
    Clips a polygon against an axis-aligned rectangular region defined by
    [left, right] × [bottom, top].

    This function performs polygon clipping by successively applying a
    half-plane clipping operation to each of the four sides of the rectangle.
    Each step trims the polygon to remain within the specified boundary.
    It is typically used to constrain texture coordinates or geometry to
    a normalized range.

    The function assumes:
      - The polygon is represented as an ordered list of (x, y) vertices.
      - The polygon may be convex or concave.
      - `clip_against_halfplane` correctly handles empty or degenerate polygons.

    :param poly: The list of (x, y) coordinates representing the polygon vertices.
    :param left: The left boundary (minimum x value).
    :param right: The right boundary (maximum x value).
    :param bottom: The bottom boundary (minimum y value).
    :param top: The top boundary (maximum y value).
    :return: A list of (x, y) vertices representing the clipped polygon.
             Returns an empty list if the polygon lies completely outside the region.
    """
    # Clip against left boundary: x >= left
    poly = _clip_against_halfplane(
        poly,
        lambda p: p[0] >= left - EPS_ABS_ZERO,
        lambda s, e: _intersect_vertical(s, e, left)
    )

    # Clip against right boundary: x <= right
    poly = _clip_against_halfplane(
        poly,
        lambda p: p[0] <= right + EPS_ABS_ZERO,
        lambda s, e: _intersect_vertical(s, e, right)
    )

    # Clip against bottom boundary: y >= bottom
    poly = _clip_against_halfplane(
        poly,
        lambda p: p[1] >= bottom - EPS_ABS_ZERO,
        lambda s, e: _intersect_horizontal(s, e, bottom)
    )

    # Clip against top boundary: y <= top
    poly = _clip_against_halfplane(
        poly,
        lambda p: p[1] <= top + EPS_ABS_ZERO,
        lambda s, e: _intersect_horizontal(s, e, top)
    )

    return poly

def get_clipped_uv_polygons(
    uv1: Tuple[float, float],
    uv2: Tuple[float, float],
    uv3: Tuple[float, float]
) -> List[Tuple[List[Tuple[float, float]], int, int]]:
    """
    Clips a UV-space triangle against all overlapping unit [0,1] × [0,1] tiles to handle texture wrapping.

    This function divides a triangle defined in UV coordinates into sub-polygons,
    each clipped to the boundaries of integer-aligned UV tiles. It is primarily used
    in texture mapping workflows where UV coordinates can extend beyond [0,1],
    and each wrapped portion must be processed or rendered separately.

    The algorithm proceeds as follows:
      1. Determine the integer tile range that the input triangle spans in both U and V.
      2. For each tile in that range, clip the triangle against the tile boundaries
         using `clip_to_unit_square()`.
      3. Clean the resulting polygon with `clean_polygon()` to remove redundant vertices.
      4. Collect all valid (non-degenerate) clipped polygons, each tagged with the
         integer tile coordinates (iu, jv) of the tile it resides in.

    The returned polygons can then be used to draw or process each wrapped portion
    of the triangle individually (e.g., in texture sampling or lightmap baking).

    :param uv1: The first vertex of the triangle in UV space (u, v).
    :param uv2: The second vertex of the triangle in UV space (u, v).
    :param uv3: The third vertex of the triangle in UV space (u, v).
    :return: A list of tuples:
             [
               (clipped_poly, iu, jv),
               ...
             ]
             where:
               - `clipped_poly` is a list of (u, v) coordinates for the clipped polygon,
               - `iu` and `jv` are the integer indices of the UV tile it belongs to.
             Returns an empty list if the triangle does not overlap any tiles.
    """
    uvs: List[Tuple[float, float]] = [uv1, uv2, uv3]

    min_u: float = min(uv[0] for uv in uvs)
    max_u: float = max(uv[0] for uv in uvs)
    min_v: float = min(uv[1] for uv in uvs)
    max_v: float = max(uv[1] for uv in uvs)

    iu_start: int = math.floor(min_u)
    iu_end: int = math.floor(max_u)
    jv_start: int = math.floor(min_v)
    jv_end: int = math.floor(max_v)

    clipped_polys: List[Tuple[List[Tuple[float, float]], int, int]] = []
    poly_base: List[Tuple[float, float]] = [uv1, uv2, uv3]

    for iu in range(iu_start, iu_end + 1):
        for jv in range(jv_start, jv_end + 1):
            # Clip triangle to current tile boundaries
            poly = clip_to_unit_square(poly_base[:], float(iu), float(iu + 1), float(jv), float(jv + 1))
            poly = clean_polygon(poly)

            # Only keep valid polygons (with area)
            if len(poly) >= 3:
                clipped_polys.append((poly, iu, jv))

    return clipped_polys

def process_uv_wrapped_face(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
    v3: Tuple[float, float, float],
    uv1: Tuple[float, float],
    uv2: Tuple[float, float],
    uv3: Tuple[float, float],
    new_vertices: List[Tuple[float, float, float]],
    new_uvs: List[Tuple[float, float]],
    new_face_triangles: List[List[int]]
) -> None:
    """
    Processes a single UV-mapped triangle, handling texture wrapping across unit UV tiles.

    This function takes a 3D triangle with associated UV coordinates and clips it
    against all overlapping [0,1]×[0,1] UV tiles (to handle texture wrapping).
    For each clipped region, it interpolates the 3D vertex positions using barycentric
    coordinates, wraps UVs into the [0,1] range, and appends new triangulated geometry
    into the provided output lists.

    The outputs (`new_vertices`, `new_uvs`, and `new_face_triangles`) are modified in place.

    The algorithm proceeds as follows:
      1. Use `get_clipped_uv_polygons()` to clip the input triangle in UV space into
         one or more polygons, one per overlapping UV tile.
      2. For each polygon:
         - Compute barycentric coordinates for each UV vertex relative to the original triangle.
         - Interpolate the corresponding 3D position.
         - Wrap the UV coordinate into the [0,1] tile.
         - Append the resulting vertex and UV to `new_vertices` and `new_uvs`.
      3. Triangulate each clipped polygon using a simple fan method
         (`(v0, vk-1, vk)` for k ≥ 2) and append indices to `new_face_triangles`.

    This function assumes:
      - The input triangle is defined in consistent UV and 3D spaces.
      - The clipped polygons are convex (as produced by `clip_to_unit_square`).
      - `barycentric_coordinates()` returns None for degenerate or invalid cases.

    :param v1: First 3D vertex position (x, y, z).
    :param v2: Second 3D vertex position (x, y, z).
    :param v3: Third 3D vertex position (x, y, z).
    :param uv1: UV coordinate corresponding to v1 (u, v).
    :param uv2: UV coordinate corresponding to v2 (u, v).
    :param uv3: UV coordinate corresponding to v3 (u, v).
    :param new_vertices: List to append the newly generated 3D vertices (modified in-place).
    :param new_uvs: List to append the wrapped UV coordinates (modified in-place).
    :param new_face_triangles: List to append triangle index lists (modified in-place).
    :return: None. The function modifies the provided lists directly.
    """
    clipped_polys = get_clipped_uv_polygons(uv1, uv2, uv3)

    for poly, iu, jv in clipped_polys:
        local_indices: List[int] = []
        skip_poly: bool = False

        # Interpolate 3D positions for each clipped UV vertex
        for p in poly:
            bary = barycentric_coordinates(p, uv1, uv2, uv3)
            if bary is None:
                skip_poly = True
                break

            b1, b2, b3 = bary

            # Interpolated 3D vertex position
            px = b1 * v1[0] + b2 * v2[0] + b3 * v3[0]
            py = b1 * v1[1] + b2 * v2[1] + b3 * v3[1]
            pz = b1 * v1[2] + b2 * v2[2] + b3 * v3[2]

            # Wrap UV into [0,1] for current tile
            nu = max(0.0, min(1.0, p[0] - iu))
            nv = max(0.0, min(1.0, p[1] - jv))

            # Append new vertex/UV and record index
            idx = len(new_vertices)
            new_vertices.append((px, py, pz))
            new_uvs.append((nu, nv))
            local_indices.append(idx)

        # Skip invalid polygons
        if skip_poly:
            for _ in range(len(local_indices)):
                new_vertices.pop()
                new_uvs.pop()
            continue

        # Triangulate clipped polygon (fan method)
        n = len(local_indices)
        if n < 3:
            continue
        for k in range(2, n):
            tri = [local_indices[0], local_indices[k - 1], local_indices[k]]
            new_face_triangles.append(tri)


