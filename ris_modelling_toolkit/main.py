# -----------------------
# Example usage
# -----------------------
import trimesh
import logging

from ris_modelling_toolkit.src.compute.model import split_mesh_for_tiled_uvs

trimesh.util.attach_to_log(level=logging.DEBUG)
mesh = trimesh.load("content/test.obj")
mesh = split_mesh_for_tiled_uvs(mesh)
mesh.export("content/test_split.obj")
