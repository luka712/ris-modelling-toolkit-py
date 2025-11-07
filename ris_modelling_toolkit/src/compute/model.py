import numpy
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

from ris_modelling_toolkit.src.data import Axis2D, UVBoundaryPointInfo
from ris_modelling_toolkit.src.util import compute_uv_boundary_points, find_uv_outside_bounds, point_in_triangle, \
    ensure_winding_order_cw


def compute_uv_boundaries(mesh: trimesh.Trimesh) -> list[UVBoundaryPointInfo]:
    """
    PRIVATE FUNCTION
    Find boundaries on the mesh where UVs cross integer boundaries outside [0,1] range.
    These boundaries can be used to split the mesh for proper UV mapping.
    1. For each triangle, check each edge's UV coordinates.
    2. Determine if the UVs cross any integer boundaries outside [0,1].
    3. If they do, calculate the corresponding 3D point on the edge.
    4. Collect all such boundaries and their point, uv information for potential mesh splitting.
    :param mesh: trimesh.Trimesh with mesh.visual.uv
    :return: The list of UVBoundaryPointInfo where splits should occur.
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


def split_mesh_at_points(mesh: trimesh.Trimesh, boundaries: list[UVBoundaryPointInfo]) -> trimesh.Trimesh:
    """
    PRIVATE FUNCTION
    Split a mesh at given points.
    :param mesh: trimesh.Trimesh to be split.
    :param boundaries: The uv based boundary points where splits should occur.
    :return: The split mesh.
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
                mesh = split_triangle(mesh, idx, boundary)
                break  # move to next point

    return mesh


def split_triangle(
        mesh: trimesh.Trimesh,
        face_index: int,
        boundary: UVBoundaryPointInfo
) -> trimesh.Trimesh:
    """
    PRIVATE FUNCTION
    Replace one triangle with three new triangles by inserting a vertex.
    Parameters:
    - mesh: trimesh.Trimesh
    - face_index: index of the face to split
    - boundary: UVBoundaryPointInfo with information about the split point
    - new_mesh: trimesh.Trimesh with updated vertices and faces
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

    is_ab_check = a == boundary.index0 and b == boundary.index1
    is_bc_check = b == boundary.index0 and c == boundary.index1
    is_ca_check = c == boundary.index0 and a == boundary.index1

    uv = None
    if boundary.uv_axis == Axis2D.X:
        x = boundary.crossed_edge

        # TODO: process X axis properly

        uv = np.array([
            x,
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
    new_faces = np.array([
        ensure_winding_order_cw([a, b, new_vertex_index], vertices),
        ensure_winding_order_cw([b, c, new_vertex_index], vertices),
        ensure_winding_order_cw([c, a, new_vertex_index], vertices)
    ])

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


def tile_mesh_uvs(mesh, sprite_index=(0, 0), sheet_size=(4, 4)):
    """
    PRIVATE FUNCTION
    Tile a mesh's UVs into [0,1] and remap to a sprite sheet cell.

    Parameters:
    - mesh: trimesh.Trimesh with mesh.visual.uv
    - sprite_index: (i, j) index of sprite cell
    - sheet_size: (columns, rows) of the sprite sheet

    Returns:
    - tiled_mesh: trimesh.Trimesh with updated vertices, faces, UVs
    """
    uvs = mesh.visual.uv.copy()
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # Step 1: Compute how many "tiles" each triangle spans
    min_uv = np.floor(np.min(uvs[faces], axis=1))
    max_uv = np.floor(np.max(uvs[faces], axis=1))
    spans = (max_uv - min_uv).astype(int)

    new_vertices = []
    new_uvs = []
    new_faces = []

    vert_map = {}  # old vertex + tile offset -> new vertex index

    for f_idx, f in enumerate(faces):
        uvs_face = uvs[f]
        # For each tile spanned by this triangle
        for du in range(spans[f_idx, 0] + 1):
            for dv in range(spans[f_idx, 1] + 1):
                # Compute new vertex indices
                new_face = []
                for vi, uv in zip(f, uvs_face):
                    key = (vi, du, dv)
                    if key not in vert_map:
                        # Duplicate vertex
                        new_vert = vertices[vi]
                        new_uv = uv - np.floor(uv)  # wrap into 0-1
                        new_vertices.append(new_vert)
                        new_uvs.append(new_uv)
                        vert_map[key] = len(new_vertices) - 1
                    new_face.append(vert_map[key])
                new_faces.append(new_face)

    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)
    new_uvs = np.array(new_uvs)

    # Step 2: Remap UVs to sprite sheet cell
    i, j = sprite_index
    cols, rows = sheet_size
    scale_u = 1.0 / cols
    scale_v = 1.0 / rows

    new_uvs[:, 0] = new_uvs[:, 0] * scale_u + i * scale_u
    new_uvs[:, 1] = new_uvs[:, 1] * scale_v + j * scale_v

    # Step 3: Build new mesh
    tiled_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    tiled_mesh.visual.uv = new_uvs

    return tiled_mesh


def tile_mesh_when_uv_out_of_bounds(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Tile a mesh's UVs into [0,1] and remap to a sprite sheet cell if UVs are out of bounds.
    :param mesh: trimesh.Trimesh with mesh.visual.uv
    :return: tiled_mesh: trimesh.Trimesh with updated vertices, faces, UVs
    """
    uv_boundaries = compute_uv_boundaries(mesh)
    split_mesh = split_mesh_at_points(mesh, uv_boundaries)
    return split_mesh
