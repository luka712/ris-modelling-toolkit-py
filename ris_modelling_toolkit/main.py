# -----------------------
# Example usage
# -----------------------
import trimesh
import logging

from ris_modelling_toolkit.src.compute.model import tile_mesh_when_uv_out_of_bounds

trimesh.util.attach_to_log(level=logging.DEBUG)
mesh = trimesh.load("content/test.obj")
mesh = tile_mesh_when_uv_out_of_bounds(mesh)
mesh.export("content/test_split.obj")
