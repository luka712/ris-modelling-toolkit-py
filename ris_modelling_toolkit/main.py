# -----------------------
# Example usage
# -----------------------
import numpy as np
import trimesh
import logging

from trimesh.visual import TextureVisuals
from trimesh.visual.material import Material, SimpleMaterial

from ris_modelling_toolkit.src.compute.model import split_geometry_for_tiled_uvs
from ris_modelling_toolkit.src.data.geometry import trimesh_mesh_to_geometry

trimesh.util.attach_to_log(level=logging.DEBUG)
mesh = trimesh.load("content/test.obj")
geometry = trimesh_mesh_to_geometry(mesh)
geometry = split_geometry_for_tiled_uvs(geometry)

faces = []

for i in range(geometry.get_triangle_count()):
    faces.append([i * 3, i * 3 + 1, i * 3 + 2])

print(geometry.vertices)
print(geometry.uvs)

new_mesh = trimesh.Trimesh(vertices=geometry.get_vertices(), faces=faces, process=False)
new_mesh.visual = TextureVisuals(material=SimpleMaterial(image=mesh.visual.material.image))
new_mesh.visual.uv = geometry.get_uv()
new_mesh.export("content/test_split.obj")




