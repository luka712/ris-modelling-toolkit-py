import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

from ris_modelling_toolkit.src.data import Axis2D, UVBoundaryIntersection
from ris_modelling_toolkit.src.util import compute_uv_boundary_points, find_uv_outside_bounds, point_in_triangle, \
    ensure_winding_order_cw
from ris_modelling_toolkit.src.util.math_util import is_valid_triangle


def _find_uv_split_boundaries(mesh: trimesh.Trimesh) -> list[UVBoundaryIntersection]:
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
     mesh : trimesh.Trimesh
         Input mesh. Requires `mesh.visual.uv` to be defined.

     Returns
     -------
     list[UVBoundaryPointInfo]
         A list of boundary split points containing both UV and 3D intersection data.
     """

    boundaries = []

    # Go through each triangle of a mesh.
    # Face is array of indices that make a triangle.
    for face in mesh.faces:

        idx_0, idx_1, idx_2 = ensure_winding_order_cw(face, mesh.vertices)

        # find positions
        pos_a = mesh.vertices[idx_0]
        pos_b = mesh.vertices[idx_1]
        pos_c = mesh.vertices[idx_2]

        # find tex coordinates
        tc_a = mesh.visual.uv[idx_0]
        tc_b = mesh.visual.uv[idx_1]
        tc_c = mesh.visual.uv[idx_2]

        # now find points to insert based on uv's and positions.

        # for each uv edge, find uv's boundaries crossed outside 0-1 range which are integer values
        # for example if uv goes from 0.8 to 1.2 we cross 1.0
        # or if uv goes from -1.5 to 3.3 we cross -1, 0, 1, 2, 3
        # do it for side a -> b, b -> c, c -> a per uv axis

        # Go from a -> b and find where coords are outside of range
        ab_x_boundary_info = find_uv_outside_bounds(idx_0, idx_1, pos_a, pos_b, tc_a, tc_b, Axis2D.X)
        ab_y_boundary_info = find_uv_outside_bounds(idx_0, idx_1, pos_a, pos_b, tc_a, tc_b, Axis2D.Y)

        # Go from b -> c
        bc_x_boundary_info = find_uv_outside_bounds(idx_1, idx_2, pos_b, pos_c, tc_b, tc_c, Axis2D.X)
        bc_y_boundary_info = find_uv_outside_bounds(idx_1, idx_2, pos_b, pos_c, tc_b, tc_c, Axis2D.Y)

        # Go from c -> a
        ca_x_boundary_info = find_uv_outside_bounds(idx_2, idx_0, pos_c, pos_a, tc_c, tc_a, Axis2D.X)
        ca_y_boundary_info = find_uv_outside_bounds(idx_2, idx_0, pos_c, pos_a, tc_c, tc_a, Axis2D.Y)

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
        for uv_boundary_point_info_list in [p_ab_x, p_ab_y, p_bc_x, p_bc_y, p_ca_x, p_ca_y]:
            if uv_boundary_point_info_list is not None:

                # Unpack list and add each point
                for uv_boundary_point_info in uv_boundary_point_info_list:
                    boundaries.append(uv_boundary_point_info)

    return boundaries


def _split_mesh_at_uv_boundaries(mesh: trimesh.Trimesh, boundaries: list[UVBoundaryIntersection]) -> trimesh.Trimesh:
    """
      Split a mesh by inserting new vertices at UV seam boundary intersection points.

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
      mesh : trimesh.Trimesh
          The input mesh to modify. Must have `mesh.visual.uv` defined.
      boundaries : list[UVBoundaryPointInfo]
          Boundary crossing descriptors, each containing:
            - 3D intersection point
            - Original edge vertex indices
            - Interpolated UV coordinate information

      Returns
      -------
      trimesh.Trimesh
          A new mesh with additional vertices and updated topology suitable
          for UV island separation or texture wrapping.

      Notes
      -----
      This function modifies topology but does not merge duplicate vertices or
      attempt to weld UV islands. A later pass may be needed depending on usage.
      """
    for boundary in boundaries:

        # Validate point length
        if len(boundary.point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        # Find exact face by barycentric test
        for idx, face in enumerate(mesh.faces):
            a, b, c = mesh.vertices[face]
            point = boundary.point
            if point_in_triangle(point, a, b, c):
                print("Splitting face", idx, "at point", point)
                mesh = _split_triangle_at_boundary(mesh, idx, boundary)
                break  # move to next point

    return mesh


def _split_triangle_at_boundary(
        mesh: trimesh.Trimesh,
        face_index: int,
        boundary: UVBoundaryIntersection
) -> trimesh.Trimesh:
    """
       Insert a new vertex into a triangle at a UV boundary intersection point and
       re-triangulate the face.

       This function handles the geometric part of UV seam splitting. When a UV
       edge crosses an integer boundary (e.g., U = 1.0 or V = 0.0), a new vertex
       must be inserted into the triangle so that UV islands can be separated
       cleanly. The new vertex lies along an original triangle edge and has its
       own UV coordinate computed from the `boundary` data.

       Steps performed:
         1. Extract the triangle to be split and determine its winding order.
         2. Compute the 3D coordinates of the new vertex (already supplied).
         3. Compute the UV coordinate for the new vertex based on the axis and
            interpolation stored in `boundary`.
         4. Append the new vertex and UV to the vertex/UV lists.
         5. Replace the one original triangle with three triangles:
               (a, b, new)  (b, c, new)  (c, a, new)
            Only triangles with valid (non-degenerate) geometry are kept.
         6. Construct and return a new trimesh.Trimesh with updated topology.

       Parameters
       ----------
       mesh : trimesh.Trimesh
           The mesh containing the triangle to split.
       face_index : int
           Index of the triangle in `mesh.faces` that should be replaced.
       boundary : UVBoundaryPointInfo
           Information about the boundary crossing:
           - `point`:    The 3D coordinate of the intersection
           - `index0`, `index1`: Original edge vertex indices
           - `uv_axis`:  Which UV axis was crossed (U or V)
           - `crossed_edge`: The integer UV value crossed (e.g., 1, 2, -1)
           - `other_coord_interpolated`: Interpolated UV on the other axis

       Returns
       -------
       trimesh.Trimesh
           A new mesh containing the updated vertices, updated UV coordinates,
           and the new triangle subdivision.

       Notes
       -----
       This function does not weld or merge resulting UV islands — it only
       performs local splitting. A later pass may group or separate islands.
       """
    faces = mesh.faces.copy()
    vertices = mesh.vertices.copy()
    uvs = mesh.visual.uv.copy()

    # Original triangle
    a, b, c = ensure_winding_order_cw(faces[face_index], vertices)
    new_vertex = boundary.point

    v0 = vertices[a]
    v1 = vertices[b]
    v2 = vertices[c]

    is_ab=(a == boundary.index0 and b == boundary.index1) or (a == boundary.index1 and b == boundary.index0)
    is_bc=(b == boundary.index0 and c == boundary.index1) or (b == boundary.index1 and c == boundary.index0)
    is_ca=(c == boundary.index0 and a == boundary.index1) or (c == boundary.index1 and a == boundary.index0)

    # TODO: ensure texture coordinates are correct!!!
    # We will need to clamp the uv's up/down depending on which edge we are working on
    uv = None
    if boundary.uv_axis == Axis2D.X:
        uv = np.array([
            boundary.crossed_edge,
            boundary.other_coord_interpolated
        ])
    else:  # Axis2D.Y
        uv = np.array([
            boundary.other_coord_interpolated,
            boundary.crossed_edge
        ])

    # Add new vertex
    new_vertex_index = len(vertices)
    vertices = np.vstack([vertices, boundary.point])
    uvs = np.vstack([uvs, uv])

    # Replace original face with new smaller faces
    face0 = ensure_winding_order_cw([a, b, new_vertex_index], vertices)
    face1 = ensure_winding_order_cw([b, c, new_vertex_index], vertices)
    face2 = ensure_winding_order_cw([c, a, new_vertex_index], vertices)

    # We only want to add valid triangles
    new_triangles = []
    if is_valid_triangle(v0, v1, new_vertex):
        new_triangles.append(face0)
    if is_valid_triangle(v1, v2, new_vertex):
        new_triangles.append(face1)
    if is_valid_triangle(v2, v0, new_vertex):
        new_triangles.append(face2)

    new_faces = np.array(new_triangles)

    # Remove original face and append new ones
    faces = np.delete(faces, face_index, axis=0)
    faces = np.vstack([faces, new_faces])

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    new_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=mesh.visual.material)
    print(type(new_mesh.visual))
    print(new_mesh.visual.uv)
    print(len(np.shape(getattr(new_mesh.visual, "uv", None))))

    return new_mesh

def split_mesh_for_tiled_uvs(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Split a mesh so that UV coordinates outside the [0, 1] range are handled correctly.

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
    mesh : trimesh.Trimesh
        Input mesh. Must have valid `mesh.visual.uv` coordinates.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with updated vertices, faces, and UVs such that no triangle
        crosses UV tile boundaries.
    """
    uv_boundaries = _find_uv_split_boundaries(mesh)
    split_mesh = _split_mesh_at_uv_boundaries(mesh, uv_boundaries)
    # TODO: do another step which will % all uv coordinates to be within 0-1 range?
    return split_mesh
