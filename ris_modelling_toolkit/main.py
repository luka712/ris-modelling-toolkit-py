# -----------------------
# Example usage
# -----------------------
import trimesh

from ris_modelling_toolkit.src.compute import compute_split_points_from_uv_overflow, split_mesh_at_points, tile_mesh_uvs

mesh = trimesh.load("content/test.obj")
split_points = compute_split_points_from_uv_overflow(mesh)
split_mesh = split_mesh_at_points(mesh, split_points)
split_mesh.export("content/test_split.obj")

sprite_index = (1, 2)  # 2nd column, 3rd row
sheet_size = (4, 4)  # 4x4 sprite sheet

tiled_mesh = tile_mesh_uvs(mesh, sprite_index, sheet_size)

tiled_mesh.export("content/mesh_tiled_sprite.obj")