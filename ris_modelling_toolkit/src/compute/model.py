import numpy as np
import trimesh

from ris_modelling_toolkit.src.data import Axis2D, UVBoundaryIntersection
from ris_modelling_toolkit.src.data.enums import WindingOrder
from ris_modelling_toolkit.src.data.geometry import Geometry
from ris_modelling_toolkit.src.data.plane import Plane
from ris_modelling_toolkit.src.data.triangle import Triangle
from ris_modelling_toolkit.src.util import compute_uv_boundary_points, find_uv_outside_bounds
from ris_modelling_toolkit.src.util.math_util import  perpendicular_vector



def _find_first_uv_split_boundary(geometry: Geometry) -> UVBoundaryIntersection | None:
    """
     Detect points on the mesh where UV coordinates cross integer tile boundaries.

     Many meshes use UVs outside the normalized [0, 1] range for tiled textures.
     When this happens, edges of triangles may cross integer UV boundaries
     (e.g., from 0.8 → 1.2 crosses U = 1). These crossings indicate where the
     mesh should be split to avoid stretched or wrapped UV interpolation.

     This function analyzes each triangle and:
       1. Checks each edge in UV space.
       2. Determines whether that edge crosses any integer U or V boundaries.
       3. Computes the corresponding 3D point on the edge where the crossing occurs.
       4. Records each such crossing as a UVBoundaryIntersection entry.

     Returns the first found boundary intersection or None if none exist.

     Parameters
     ----------
     geometry : Geometry containing vertices and UV coordinates.

     Returns
     -------
     UVBoundaryPointInfo
         A  boundary split point containing both UV and 3D intersection data.
     """

    boundaries = []

    triangle_count = geometry.get_triangle_count()

    # Go through each triangle of a mesh.
    # Face is array of indices that make a triangle.
    for i in range(triangle_count):

        triangle = geometry.get_triangle(i, WindingOrder.CCW)

        # find positions
        pos_a = triangle.v0
        pos_b = triangle.v1
        pos_c = triangle.v2

        # find tex coordinates
        tc_a = triangle.uv0
        tc_b = triangle.uv1
        tc_c = triangle.uv2

        # now find points to insert based on uv's and positions.

        # for each uv edge, find uv's boundaries crossed outside 0-1 range which are integer values
        # for example if uv goes from 0.8 to 1.2 we cross 1.0
        # or if uv goes from -1.5 to 3.3 we cross -1, 0, 1, 2, 3
        # do it for side a -> b, b -> c, c -> a per uv axis

        # Go from a -> b and find where coords are outside of range for x
        ab_x_boundary_info = find_uv_outside_bounds(pos_a, pos_b, tc_a, tc_b, Axis2D.X)
        if ab_x_boundary_info is not None:
            p_ab_x = compute_uv_boundary_points(ab_x_boundary_info)
            if p_ab_x is not None and len(p_ab_x) > 0:
                return p_ab_x[0]

        # Go from b -> c for x
        bc_x_boundary_info = find_uv_outside_bounds(pos_b, pos_c, tc_b, tc_c, Axis2D.X)
        if bc_x_boundary_info is not None:
            p_bc_x = compute_uv_boundary_points(bc_x_boundary_info)
            if p_bc_x is not None and len(p_bc_x) > 0:
                return p_bc_x[0]

        # Go from c -> a for x
        ca_x_boundary_info = find_uv_outside_bounds(pos_c, pos_a, tc_c, tc_a, Axis2D.X)
        if ca_x_boundary_info is not None:
            p_ca_x = compute_uv_boundary_points(ca_x_boundary_info)
            if p_ca_x is not None and len(p_ca_x) > 0:
                return p_ca_x[0]

        # Now do the same for y axis
        # Go from a -> b and find where coords are outside of range for y
        ab_y_boundary_info = find_uv_outside_bounds(pos_a, pos_b, tc_a, tc_b, Axis2D.Y)
        if ab_y_boundary_info is not None:
            p_ab_y = compute_uv_boundary_points(ab_y_boundary_info)
            if p_ab_y is not None and len(p_ab_y) > 0:
                return p_ab_y[0]

        # Go from b -> c for y
        bc_y_boundary_info = find_uv_outside_bounds(pos_b, pos_c, tc_b, tc_c, Axis2D.Y)
        if bc_y_boundary_info is not None:
            p_bc_y = compute_uv_boundary_points(bc_y_boundary_info)
            if p_bc_y is not None and len(p_bc_y) > 0:
                return p_bc_y[0]

        # Go from c -> a for y
        ca_y_boundary_info = find_uv_outside_bounds(pos_c, pos_a, tc_c, tc_a, Axis2D.Y)
        if ca_y_boundary_info is not None:
            p_ca_y = compute_uv_boundary_points(ca_y_boundary_info)
            if p_ca_y is not None and len(p_ca_y) > 0:
                return p_ca_y[0]

    return None

def _find_uv_split_boundaries(geometry: Geometry) -> list[UVBoundaryIntersection]:
    """
     Detect points on the mesh where UV coordinates cross integer tile boundaries.

     Many meshes use UVs outside the normalized [0, 1] range for tiled textures.
     When this happens, edges of triangles may cross integer UV boundaries
     (e.g., from 0.8 → 1.2 crosses U = 1). These crossings indicate where the
     mesh should be split to avoid stretched or wrapped UV interpolation.

     This function analyzes each triangle and:
       1. Checks each edge in UV space.
       2. Determines whether that edge crosses any integer U or V boundaries.
       3. Computes the corresponding 3D point on the edge where the crossing occurs.
       4. Records each such crossing as a UVBoundaryIntersection entry.

     The resulting list can be used to:
       - Create additional vertices where UV seams must exist.
       - Properly unwrap or duplicate geometry for tiled UVs.
       - Avoid texture interpolation artifacts across tile boundaries.

     Parameters
     ----------
     geometry : Geometry containing vertices and UV coordinates.

     Returns
     -------
     list[UVBoundaryPointInfo]
         A list of boundary split points containing both UV and 3D intersection data.
     """

    boundaries = []

    triangle_count = geometry.get_triangle_count()

    # Go through each triangle of a mesh.
    # Face is array of indices that make a triangle.
    for i in range(triangle_count):

        triangle = geometry.get_triangle(i, WindingOrder.CCW)

        # find positions
        pos_a = triangle.v0
        pos_b = triangle.v1
        pos_c = triangle.v2

        # find tex coordinates
        tc_a = triangle.uv0
        tc_b = triangle.uv1
        tc_c = triangle.uv2

        # now find points to insert based on uv's and positions.

        # for each uv edge, find uv's boundaries crossed outside 0-1 range which are integer values
        # for example if uv goes from 0.8 to 1.2 we cross 1.0
        # or if uv goes from -1.5 to 3.3 we cross -1, 0, 1, 2, 3
        # do it for side a -> b, b -> c, c -> a per uv axis

        # Go from a -> b and find where coords are outside of range
        ab_x_boundary_info = find_uv_outside_bounds(pos_a, pos_b, tc_a, tc_b, Axis2D.X)
        ab_y_boundary_info = find_uv_outside_bounds(pos_a, pos_b, tc_a, tc_b, Axis2D.Y)

        # Go from b -> c
        bc_x_boundary_info = find_uv_outside_bounds(pos_b, pos_c, tc_b, tc_c, Axis2D.X)
        bc_y_boundary_info = find_uv_outside_bounds(pos_b, pos_c, tc_b, tc_c, Axis2D.Y)

        # Go from c -> a
        ca_x_boundary_info = find_uv_outside_bounds(pos_c, pos_a, tc_c, tc_a, Axis2D.X)
        ca_y_boundary_info = find_uv_outside_bounds(pos_c, pos_a, tc_c, tc_a, Axis2D.Y)

        # Now from those boundaries find the 3D points on the edges
        # For example if we cross 1.0 on U axis from 0.8 to 1.2 we find the point on the edge a -> b where U is 1.0

        # Find points where uv's cross those boundaries
        # Result for each is list of UVBoundaryPointInfo or None
        p_ab_x = compute_uv_boundary_points(ab_x_boundary_info)
        p_ab_y = compute_uv_boundary_points(ab_y_boundary_info)

        p_bc_x = compute_uv_boundary_points(bc_x_boundary_info)
        p_bc_y = compute_uv_boundary_points(bc_y_boundary_info)

        p_ca_x = compute_uv_boundary_points(ca_x_boundary_info)
        p_ca_y = compute_uv_boundary_points(ca_y_boundary_info)

        # Go through all boundary points and add them to the list
        # Note that each p_ab_x...p_ca_y is a list of UVBoundaryPointInfo or None
        for uv_boundary_point_info_list in [p_ab_x, p_ab_x, p_ab_y, p_bc_y, p_ca_x, p_ca_y]:
            if uv_boundary_point_info_list is not None:

                # Unpack list and add each point
                for uv_boundary_point_info in uv_boundary_point_info_list:
                    boundaries.append(uv_boundary_point_info)

    return boundaries


def _split_geometry_at_uv_boundaries(geometry: Geometry, boundaries: list[UVBoundaryIntersection]) -> Geometry:
    """
      Split a geometry by inserting new vertices at UV seam boundary intersection points.

      When UV coordinates cross integer tile boundaries, the mesh requires
      additional vertices so that each UV island can be separated cleanly.
      This function takes the UV boundary intersection data produced by
      `compute_uv_split_points` and applies the corresponding geometric splits.

      For each boundary point:
        - The triangle containing the 3D point is located via barycentric testing.
        - That triangle is replaced with three triangles sharing a new vertex.
        - UV coordinates for the new vertex are assigned based on interpolation.

      Parameters
      ----------
      geometry : Geometry
          The input geometry to modify.
      boundaries : list[UVBoundaryPointInfo]
          Boundary crossing descriptors, each containing:
            - 3D intersection point
            - Original edge vertex indices
            - Interpolated UV coordinate information

      Returns
      -------
      Geometry
          A new geometry with additional vertices and updated topology suitable
          for UV island separation or texture wrapping.

      Notes
      -----
      This function modifies topology but does not merge duplicate vertices or
      attempt to weld UV islands. A later pass may be needed depending on usage.
      """

    triangle_count = geometry.get_triangle_count()
    new_triangles = []

    for boundary in boundaries:

        # Validate point length
        if len(boundary.point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        # Go through each triangle and find which triangle contains this point
        for idx in range(triangle_count):
            triangle = geometry.get_triangle(idx, WindingOrder.CCW)
            point = boundary.point
            is_on_edge, line_segment = triangle.point_on_triangle_edge(point)
            if is_on_edge:
                print("Splitting face", idx, "at point", point)
                new_split_triangles = _split_triangle_at_boundary(triangle, boundary)
                new_triangles.extend(new_split_triangles)
                break  # move to next point

    # Now create new geometry from new triangles
    vertices = []
    uvs = []
    for tri in new_triangles:
        vertices.extend([tri.v0, tri.v1, tri.v2])
        uvs.extend([tri.uv0, tri.uv1, tri.uv2])
    return Geometry(
        vertices=np.array(vertices),
        uvs=np.array(uvs)
    )



def _split_geometry_at_uv_boundary(geometry: Geometry, boundary: UVBoundaryIntersection) -> Geometry:
    """
      Split a geometry by inserting new vertices at UV seam boundary intersection points.

      When UV coordinates cross integer tile boundaries, the mesh requires
      additional vertices so that each UV island can be separated cleanly.
      This function takes the UV boundary intersection data produced by
      `compute_uv_split_points` and applies the corresponding geometric splits.

      For each boundary point:
        - The triangle containing the 3D point is located via barycentric testing.
        - That triangle is replaced with three triangles sharing a new vertex.
        - UV coordinates for the new vertex are assigned based on interpolation.

      Parameters
      ----------
      geometry : Geometry
          The input geometry to modify.
      boundary : UVBoundaryPointInfo
          Boundary crossing description, each containing:
            - 3D intersection point
            - Original edge vertex indices
            - Interpolated UV coordinate information

      Returns
      -------
      Geometry
          A new geometry with additional vertices and updated topology suitable
          for UV island separation or texture wrapping.

      Notes
      -----
      This function modifies topology but does not merge duplicate vertices or
      attempt to weld UV islands. A later pass may be needed depending on usage.
      """

    triangle_count = geometry.get_triangle_count()
    new_triangles = []

    # Validate point length
    if len(boundary.point) != 3:
         raise ValueError("Point must be a 3D coordinate.")

    # Go through each triangle and find which triangle contains this point
    for idx in range(triangle_count):
        triangle = geometry.get_triangle(idx, WindingOrder.CCW)
        point = boundary.point
        is_on_edge, line_segment = triangle.point_on_triangle_edge(point)
        if is_on_edge:
            print("Splitting face", idx, "at point", point)
            new_split_triangles = _split_triangle_at_boundary(triangle, boundary)
            new_triangles.extend(new_split_triangles)
            geometry.remove_triangle(idx)
            break  # move to next point

    # Now create new geometry from new triangles
    vertices = []
    uvs = []
    for tri in new_triangles:
        vertices.extend([tri.v0, tri.v1, tri.v2])
        uvs.extend([tri.uv0, tri.uv1, tri.uv2])


    return Geometry(
        vertices=np.concatenate((geometry.vertices, np.array(vertices)), axis=0),
        uvs=np.concatenate((geometry.uvs, np.array(uvs)), axis=0)
    )


def _split_triangle_at_boundary(
        triangle: Triangle,
        boundary: UVBoundaryIntersection
) -> list[Triangle]:
    # We have line segment that contains point.
    # Now we need to split triangle by perpendicular plane that goes through that point
    # and is aligned with that line segment
    # We can imagine this as cutting the triangle in half along that line segment
    p_vector = perpendicular_vector(triangle.normal())
    perpendicular_plane_0 = Plane(boundary.point, p_vector * -1)
    perpendicular_plane_1 = Plane(boundary.point, p_vector)

    # Clip triangle with that plane
    # Result will be two triangles on each side of the plane
    # Each triangle will have one vertex as the boundary point
    # and two original triangle vertices
    # We need to create new UV coordinates for the boundary point
    # based on the side of the plane it is on
    clipped_triangles_0 = triangle.clip_by_plane(perpendicular_plane_0)
    clipped_triangles_1 = triangle.clip_by_plane(perpendicular_plane_1)

    if clipped_triangles_0 is None and clipped_triangles_1 is None:
        return triangle.copy()

    # on each clipped triangle, set the uv for the boundary point
    for t in clipped_triangles_0:
        t.uv0 = set_new_uv(t.v0, t.uv0, boundary)
        t.uv1 = set_new_uv(t.v1, t.uv1, boundary)
        t.uv2 = set_new_uv(t.v2, t.uv2, boundary)

    for t in clipped_triangles_1:
        t.uv0 = set_new_uv(t.v0, t.uv0, boundary)
        t.uv1 = set_new_uv(t.v1, t.uv1, boundary)
        t.uv2 = set_new_uv(t.v2, t.uv2, boundary)

    triangles = []
    for t in (clipped_triangles_0 + clipped_triangles_1):
        if t.is_valid_triangle():
            triangles.append(t)

    return triangles

def set_new_uv(vertex: np.ndarray[list[float]], vertex_uv: np.ndarray[list[float]], boundary: UVBoundaryIntersection):
    """
    If the vertex matches the boundary point, set its UV coordinates based on the boundary info and return updated UVs.
    :param vertex: The vertex of triangle to check and potentially update.
    :param vertex_uv: The UV coordinates of the vertex.
    :param boundary: The UV boundary intersection info.
    :return: The updated UV coordinates.
    """

    if len(vertex) != 3:
        raise ValueError("Vertex must be a 3D coordinate.")

    if len(vertex_uv) != 2:
        raise ValueError("Vertex UV must be a 2D coordinate.")

    if np.allclose(vertex,  boundary.point):
        if boundary.uv_axis == Axis2D.X:
            crossed_edge = 1
            if vertex_uv[0] <= boundary.crossed_edge:
                crossed_edge = 0

            return np.vstack([crossed_edge, boundary.other_coord_interpolated]).reshape(-1)
        else:
            crossed_edge = 1
            if vertex_uv[1] <= boundary.crossed_edge:
                crossed_edge = 0

            return  np.vstack([boundary.other_coord_interpolated,crossed_edge]).reshape(-1)

    return vertex_uv


def split_geometry_for_tiled_uvs(geometry: Geometry) -> Geometry:
    """
    Split a geometry so that UV coordinates outside the [0, 1] range are handled correctly.

    Some meshes use UV coordinates larger than 1.0 or less than 0.0 to represent
    tiled or repeated textures. In such cases, a single triangle may span across
    multiple UV tiles, which causes incorrect texture interpolation.

    This function:
      1. Detects where triangle edges cross integer UV boundaries (tile edges).
      2. Inserts new vertices at those crossing points.
      3. Splits triangles accordingly so each resulting triangle lies fully
         inside a single UV tile.

    The resulting mesh has duplicated vertices where necessary, making it
    suitable for:
      - Proper GPU texture sampling
      - Lightmap baking
      - Texture atlas packing
      - Export to game engines that do not support tiled UV interpolation

    Parameters
    ----------
    geometry : Geometry
        Input geometry with vertices and UV coordinates.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with updated vertices, faces, and UVs such that no triangle
        crosses UV tile boundaries.
    """
    uv_boundary = _find_first_uv_split_boundary(geometry)

    if uv_boundary is None:
        return geometry

    # WE SHOULD SPLIT TRIANGLE INSTEAD OF GEOMETRY MULTIPLE TIMES
    # a) FIND BOUNDARIES ON TRIANGLE
    # b) SPLIT THAT TRIANGLE BY BOUNDARIES AND NORMALIZE UVS ON NEW TRIANGLES
    # c) REPEAT UNTIL NO MORE BOUNDARIES ARE FOUND
    # d) ONLY THEN REMOVE ORIGINAL TRIANGLE
    # e) FROM NEW TRIANGLES REMOVE DUPES AND ADD NEW TRIANGLES TO GEOMETRY
    # f) CONTINUE TO NEXT TRIANGLE


    # Now split geometry at each found uv boundary until none are left
    # First split is here so that we do not modify original geometry in loop
    split_geometry = _split_geometry_at_uv_boundary(geometry, uv_boundary)
    split_geometry.print_triangle_info()
    uv_boundary = _find_first_uv_split_boundary(split_geometry)

    # We need to repeat until no more boundaries are found
    while uv_boundary is not None:
            split_geometry = _split_geometry_at_uv_boundary(split_geometry, uv_boundary)
            split_geometry.print_triangle_info()
            uv_boundary = _find_first_uv_split_boundary(split_geometry)

    return split_geometry
