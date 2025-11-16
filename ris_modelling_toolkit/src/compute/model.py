from typing import List, Tuple

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from ris_modelling_toolkit.src.compute.polygon_2d import process_uv_wrapped_face


def generate_uv_wrapped_obj(
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



def _export_to_obj_string(
        new_vertices: List[Tuple[float, float, float]],
        new_uvs: List[Tuple[float, float]],
        new_faces: List[List[int]],
        obj_name: str,
        mtl_lib: str,
        material: str
) -> trimesh:
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

    return mesh

def uv_wrap_obj_model(input: str, output: str) -> bool:
    """
    UV wrap an OBJ model from input file and save to output file.
    :param input: The input OBJ file path.
    :param output: The output OBJ file path.
    :return: True if successful, False otherwise.
    """
    # Load the mesh from file
    loaded = trimesh.load(input)

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

            sub_vertices, sub_uvs, sub_faces = generate_uv_wrapped_obj(geom.vertices, geom.faces, uvs_per_face)

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
            all_new_vertices, all_new_uvs, all_new_faces = generate_uv_wrapped_obj(mesh.vertices, mesh.faces,
                                                                                   uvs_per_face)
        else:
            raise ValueError("No UVs or faces found in the mesh")

    obj_mesh = _export_to_obj_string(all_new_vertices, all_new_uvs, all_new_faces, obj_name, mtl_lib, material)
    # Export to bytes
    mesh.export(output)
    return True
