import math
import io
import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial
from typing import Tuple, List, Optional, Any

# Numerical tolerances
EPS_ABS_ZERO = 1e-9  # For absolute near-zero checks (e.g., areas, distances)
EPS_REL_VALID = 1e-6  # For relative validity checks (e.g., barycentric sums, point equality)


def signed_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Compute twice the signed area of a triangle in 2D.

    :param p1: The first 2D point (u, v).
    :param p2: The second 2D point (u, v).
    :param p3: The third 2D point (u, v).
    :return: Twice the signed area of the triangle formed by the points.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])


def barycentric_coordinates(target: Tuple[float, float], uv1: Tuple[float, float], uv2: Tuple[float, float],
                            uv3: Tuple[float, float]) -> Optional[Tuple[float, float, float]]:
    """
    Compute barycentric coordinates of target point within the UV triangle.

    :param target: The 2D point (u, v) for which to compute barycentric coordinates.
    :param uv1: The first UV coordinate of the triangle (u, v).
    :param uv2: The second UV coordinate of the triangle (u, v).
    :param uv3: The third UV coordinate of the triangle (u, v).
    :return: A tuple of barycentric coordinates (b1, b2, b3) if the point is inside the triangle, else None.
    """
    a0 = signed_area(uv1, uv2, uv3)
    if abs(a0) < EPS_ABS_ZERO:
        return None
    a1 = signed_area(target, uv2, uv3)
    a2 = signed_area(uv1, target, uv3)
    a3 = signed_area(uv1, uv2, target)
    b1, b2, b3 = a1 / a0, a2 / a0, a3 / a0
    if abs(b1 + b2 + b3 - 1.0) > EPS_REL_VALID or min(b1, b2, b3) < -EPS_REL_VALID:
        return None
    return (b1, b2, b3)


def clip_against_halfplane(poly: List[Tuple[float, float]], inside_func: callable, intersect_func: callable) -> List[
    Tuple[float, float]]:
    """
    Clip a polygon against a halfplane defined by inside_func and intersect_func.

    :param poly: The input list of 2D points forming the polygon.
    :param inside_func: A callable that takes a point and returns True if inside the halfplane.
    :param intersect_func: A callable that takes two points (start, end) and returns the intersection point or None.
    :return: The clipped polygon as a list of 2D points.
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
                inter = intersect_func(prev_point, curr_point)
                if inter is not None:
                    output.append(inter)
            output.append(curr_point)
        elif prev_inside:
            inter = intersect_func(prev_point, curr_point)
            if inter is not None:
                output.append(inter)
        prev_point = curr_point
        prev_inside = curr_inside
    return output


def clean_polygon(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Remove duplicate consecutive points from a polygon and ensure it's closed properly.

    :param poly: The input list of 2D points forming the polygon.
    :return: The cleaned polygon as a list of 2D points, or empty list if fewer than 3 points.
    """
    if len(poly) < 3:
        return []
    cleaned: List[Tuple[float, float]] = []
    prev_p: Optional[Tuple[float, float]] = None
    for p in poly:
        if prev_p is None or (abs(p[0] - prev_p[0]) > EPS_REL_VALID or abs(p[1] - prev_p[1]) > EPS_REL_VALID):
            cleaned.append(p)
            prev_p = p
    # Remove last if it matches first
    if len(cleaned) > 2 and abs(cleaned[0][0] - cleaned[-1][0]) < EPS_REL_VALID and abs(
            cleaned[0][1] - cleaned[-1][1]) < EPS_REL_VALID:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def intersect_vertical(s: Tuple[float, float], e: Tuple[float, float], u_val: float) -> Optional[Tuple[float, float]]:
    """
    Compute intersection with vertical line u = u_val.

    :param s: The start 2D point (u, v).
    :param e: The end 2D point (u, v).
    :param u_val: The u-value of the vertical line.
    :return: The intersection point (u_val, v_inter) if it exists within the segment, else None.
    """
    du = e[0] - s[0]
    if abs(du) < EPS_ABS_ZERO:
        return None
    t = (u_val - s[0]) / du
    if -EPS_ABS_ZERO <= t <= 1 + EPS_ABS_ZERO:
        return (u_val, s[1] + t * (e[1] - s[1]))
    return None


def intersect_horizontal(s: Tuple[float, float], e: Tuple[float, float], v_val: float) -> Optional[Tuple[float, float]]:
    """
    Compute intersection with horizontal line v = v_val.

    :param s: The start 2D point (u, v).
    :param e: The end 2D point (u, v).
    :param v_val: The v-value of the horizontal line.
    :return: The intersection point (u_inter, v_val) if it exists within the segment, else None.
    """
    dv = e[1] - s[1]
    if abs(dv) < EPS_ABS_ZERO:
        return None
    t = (v_val - s[1]) / dv
    if -EPS_ABS_ZERO <= t <= 1 + EPS_ABS_ZERO:
        return (s[0] + t * (e[0] - s[0]), v_val)
    return None


def clip_to_unit_square(poly: List[Tuple[float, float]], left: float, right: float, bottom: float, top: float) -> List[
    Tuple[float, float]]:
    """
    Clip a polygon to the unit square [left, right] x [bottom, top].

    :param poly: The input list of 2D points forming the polygon.
    :param left: The left boundary of the square (u-min).
    :param right: The right boundary of the square (u-max).
    :param bottom: The bottom boundary of the square (v-min).
    :param top: The top boundary of the square (v-max).
    :return: The clipped polygon as a list of 2D points.
    """
    # Clip against u >= left
    poly = clip_against_halfplane(poly, lambda p: p[0] >= left - EPS_ABS_ZERO,
                                  lambda s, e: intersect_vertical(s, e, left))
    # Clip against u <= right
    poly = clip_against_halfplane(poly, lambda p: p[0] <= right + EPS_ABS_ZERO,
                                  lambda s, e: intersect_vertical(s, e, right))
    # Clip against v >= bottom
    poly = clip_against_halfplane(poly, lambda p: p[1] >= bottom - EPS_ABS_ZERO,
                                  lambda s, e: intersect_horizontal(s, e, bottom))
    # Clip against v <= top
    poly = clip_against_halfplane(poly, lambda p: p[1] <= top + EPS_ABS_ZERO,
                                  lambda s, e: intersect_horizontal(s, e, top))
    return poly


def get_clipped_uv_polygons(uv1: Tuple[float, float], uv2: Tuple[float, float], uv3: Tuple[float, float]) -> List[
    Tuple[List[Tuple[float, float]], int, int]]:
    """
    Clip the UV triangle against all overlapping unit tiles for wrapping.

    :param uv1: The first UV coordinate of the triangle (u, v).
    :param uv2: The second UV coordinate of the triangle (u, v).
    :param uv3: The third UV coordinate of the triangle (u, v).
    :return: A list of tuples (clipped_poly, iu, jv) where clipped_poly is the list of points in the tile,
             iu and jv are the integer tile coordinates.
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
            poly = clip_to_unit_square(poly_base[:], float(iu), float(iu + 1), float(jv), float(jv + 1))
            poly = clean_polygon(poly)
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
    Process a single face: clip UVs, generate new vertices/UVs, and triangulate.

    This function modifies the input lists in-place by appending new data.

    :param v1: The first 3D vertex position (x, y, z).
    :param v2: The second 3D vertex position (x, y, z).
    :param v3: The third 3D vertex position (x, y, z).
    :param uv1: The first UV coordinate (u, v).
    :param uv2: The second UV coordinate (u, v).
    :param uv3: The third UV coordinate (u, v).
    :param new_vertices: List to append new interpolated 3D vertices.
    :param new_uvs: List to append new wrapped UV coordinates.
    :param new_face_triangles: List to append new triangle face indices.
    :return: None (modifies inputs in-place).
    """
    clipped_polys = get_clipped_uv_polygons(uv1, uv2, uv3)
    for poly, iu, jv in clipped_polys:
        # Compute barycentrics and interpolate for each point in the clipped poly
        local_indices: List[int] = []
        skip_poly: bool = False
        for p in poly:
            bary = barycentric_coordinates(p, uv1, uv2, uv3)
            if bary is None:
                skip_poly = True
                break
            b1, b2, b3 = bary
            # Interpolate vertex position
            px = b1 * v1[0] + b2 * v2[0] + b3 * v3[0]
            py = b1 * v1[1] + b2 * v2[1] + b3 * v3[1]
            pz = b1 * v1[2] + b2 * v2[2] + b3 * v3[2]
            # Wrap UV to [0,1] for this tile
            nu = p[0] - iu
            nv = p[1] - jv
            # Clamp for numerical safety
            nu = max(0.0, min(1.0, nu))
            nv = max(0.0, min(1.0, nv))
            # Append new vertex and UV
            idx = len(new_vertices)
            new_vertices.append((px, py, pz))
            new_uvs.append((nu, nv))
            local_indices.append(idx)
        if skip_poly:
            # Remove any partially added points
            for _ in range(len(local_indices)):
                new_vertices.pop()
                new_uvs.pop()
            continue
        # Triangulate the polygon using fan method (assumes convex)
        n = len(local_indices)
        if n < 3:
            continue
        for k in range(2, n):
            tri = [local_indices[0], local_indices[k - 1], local_indices[k]]
            new_face_triangles.append(tri)


def generate_wrapped_obj(
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs_per_face: np.ndarray
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], List[List[int]]]:
    """
    Generate new vertices, UVs, and faces for all input faces with UV wrapping.

    :param vertices: NumPy array of original 3D vertices (n_v, 3).
    :param faces: NumPy array of face indices (n_f, 3).
    :param uvs_per_face: NumPy array of UVs per face corner (n_f, 3, 2).
    :return: A tuple of (new_vertices, new_uvs, new_face_triangles) where:
             - new_vertices: List of new interpolated 3D vertices.
             - new_uvs: List of new wrapped UV coordinates in [0,1].
             - new_face_triangles: List of triangle indices [i1, i2, i3].
    """
    new_vertices: List[Tuple[float, float, float]] = []
    new_uvs: List[Tuple[float, float]] = []
    new_face_triangles: List[List[int]] = []
    for fi, face in enumerate(faces):
        v_idxs = face
        v1 = tuple(vertices[int(v_idxs[0])])
        v2 = tuple(vertices[int(v_idxs[1])])
        v3 = tuple(vertices[int(v_idxs[2])])
        uvs_f = uvs_per_face[fi]
        uv1 = tuple(uvs_f[0])
        uv2 = tuple(uvs_f[1])
        uv3 = tuple(uvs_f[2])
        process_uv_wrapped_face(v1, v2, v3, uv1, uv2, uv3, new_vertices, new_uvs, new_face_triangles)
    return new_vertices, new_uvs, new_face_triangles


def export_to_obj_string(
        new_vertices: List[Tuple[float, float, float]],
        new_uvs: List[Tuple[float, float]],
        new_faces: List[List[int]],
        obj_name: str,
        mtl_lib: str,
        material: str
) -> str:
    """
    Export the processed mesh to OBJ string using trimesh.

    :param new_vertices: List of new 3D vertices (x, y, z).
    :param new_uvs: List of new UV coordinates (u, v).
    :param new_faces: List of triangle indices [i1, i2, i3].
    :param obj_name: The base object name (will append '_wrapped').
    :param mtl_lib: The MTL library filename to insert.
    :param material: The material name to insert.
    :return: The OBJ content as a string, or a comment if no geometry.
    """
    if not new_vertices or not new_faces:
        return "# No geometry generated\n"

    vertices_array = np.array(new_vertices)
    faces_array = np.array(new_faces)
    uvs_array = np.array(new_uvs)

    mesh = trimesh.Trimesh(
        vertices=vertices_array,
        faces=faces_array,
        visual=TextureVisuals(
            uv=uvs_array,
            material=SimpleMaterial(name=material)
        ),
        name=obj_name + '_wrapped'
    )

    # Export to bytes
    mesh.export("content/wad.obj")



# Example usage loading from file
if __name__ == "__main__":
    # Load the mesh from file
    loaded = trimesh.load("content/test.obj")

    # Handle Scene or single Trimesh
    all_new_vertices: List[Tuple[float, float, float]] = []
    all_new_uvs: List[Tuple[float, float]] = []
    all_new_faces: List[List[int]] = []
    mtl_lib = 'material.mtl'  # From your OBJ header

    if isinstance(loaded, trimesh.Scene):
        print("Loaded as Scene (multi-object OBJ). Processing all sub-geometries with UVs...")
        processed_count = 0
        for geom_name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            if not (hasattr(geom.visual, 'uv') and geom.visual.uv is not None):
                print(f"Skipping {geom_name}: No UVs")
                continue

            # Safe material extraction
            if (hasattr(geom, 'visual') and geom.visual and
                    hasattr(geom.visual, 'material') and geom.visual.material):
                mat_name = getattr(geom.visual.material, 'name', 'default')
            else:
                mat_name = 'default'  # Or parse from usemtl if needed

            print(f"Processing {geom_name} with material '{mat_name}'")

            # Offset faces for this sub-mesh
            offset = len(all_new_vertices)

            # FIXED: Use indexing to expand per-vertex UVs to per-face-corner
            uvs_per_face = geom.visual.uv[geom.faces]

            sub_vertices, sub_uvs, sub_faces = generate_wrapped_obj(geom.vertices, geom.faces, uvs_per_face)

            all_new_vertices.extend(sub_vertices)
            all_new_uvs.extend(sub_uvs)
            all_new_faces.extend([[i + offset for i in f] for f in sub_faces])

            processed_count += 1

        if processed_count == 0:
            raise ValueError("No sub-geometries with UVs found in the Scene")

        obj_name = 'pwad_freedoom'  # Base name for output
        material = 'default'  # Or aggregate from sub-mats if needed

    else:
        # Single Trimesh case
        print("Loaded as single Trimesh.")
        mesh = loaded
        obj_name = getattr(mesh, 'name', 'pwad_freedoom')

        # Safe material extraction
        if (hasattr(mesh, 'visual') and mesh.visual and
                hasattr(mesh.visual, 'material') and mesh.visual.material):
            material = getattr(mesh.visual.material, 'name', 'default')
        else:
            material = 'default'

        if len(mesh.faces) > 0 and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            # FIXED: Use indexing to expand per-vertex UVs to per-face-corner
            uvs_per_face = mesh.visual.uv[mesh.faces]
            all_new_vertices, all_new_uvs, all_new_faces = generate_wrapped_obj(mesh.vertices, mesh.faces, uvs_per_face)
        else:
            raise ValueError("No UVs or faces found in the mesh")

    output_obj = export_to_obj_string(all_new_vertices, all_new_uvs, all_new_faces, obj_name, mtl_lib, material)
    print(output_obj)

    # Optionally, save to file
    with open("content/test_split.obj", "w") as f:
        f.write(output_obj)
    print("Exported to content/test_split.obj")