import numpy as np
import trimesh

from ris_modelling_toolkit.src.data.enums import WindingOrder
from ris_modelling_toolkit.src.data.triangle import Triangle
from ris_modelling_toolkit.src.util.winding_order_util import ensure_winding_order_triangle

class Geometry:
    """
    This class represents a 3D geometry with vertices and UV coordinates.
    :param vertices: A numpy array of floats representing the vertices of the geometry.
    :param uvs: A numpy array of floats representing the UV coordinates of the geometry.
    """
    def __init__(self, vertices: np.ndarray[tuple[float]], uvs: np.ndarray[tuple[float]]):
        """
        Initialize the Geometry with vertices and UV coordinates.
        :param vertices: The vertices of the geometry.
        :param uvs: The UV coordinates of the geometry.
        """
        self.vertices = vertices
        self.uvs = uvs
        self._triangles = None

    def get_vertices(self) -> np.ndarray[tuple[float]]:
        """
        Get the vertices of the geometry.
        :return: The vertices as a numpy array of floats.
        """
        return self.vertices

    def get_uv(self) -> np.ndarray[tuple[float]]:
        """
        Get the UV coordinates of the geometry.
        :return: Returns the UV coordinates as a numpy array of floats.
        """
        return self.uvs

    def get_triangle_count(self) -> int:
        """
        Get the number of triangles in the geometry.
        :return: The number of triangles.
        """
        return len(self.vertices) // 3

    def remove_dupe_triangles(self) -> "Geometry":
        """
        Remove duplicate triangles from self.
        :return: The reference to self with duplicate triangles removed.
        """
        unique_triangles = []

        # Go through each triangle of this geometry
        for i in range(self.get_triangle_count()):
            triangle = self.get_triangle(i)
            has_dupe = False
            for ut in unique_triangles:
                if ut.equal(triangle):
                    has_dupe = True
            if not has_dupe:
                unique_triangles.append(triangle)

        new_vertices = []
        new_uvs = []
        for tri in unique_triangles:
            new_vertices.append(tri.v0)
            new_vertices.append(tri.v1)
            new_vertices.append(tri.v2)
            new_uvs.append(tri.uv0)
            new_uvs.append(tri.uv1)
            new_uvs.append(tri.uv2)

        self.vertices =  np.vstack(new_vertices)
        self.uvs = np.vstack(new_uvs)
        return self


    def get_triangle(self, idx, winding_order: WindingOrder | None = None) -> Triangle:
        """
        Get the triangle at the specified index.
        :param idx: The index of the triangle to retrieve.
        :param winding_order: Whether the triangle should be in counter-clockwise order.
        :return: A Triangle object representing the triangle at the specified index.
        """
        if idx < 0 or idx >= self.get_triangle_count():
            raise IndexError("Triangle index out of range.")

        v0 = self.vertices[idx * 3]
        v1 = self.vertices[idx * 3 + 1]
        v2 = self.vertices[idx * 3 + 2]

        pack_vertices = [v0, v1, v2]

        pack_uvs = [
            self.uvs[idx * 3],
            self.uvs[idx * 3 + 1],
            self.uvs[idx * 3 + 2]
        ]

        indices = [0, 1, 2]

        if winding_order == WindingOrder.CCW:
            indices = ensure_winding_order_triangle(v0, v1,v2, WindingOrder.CCW)
        elif winding_order == WindingOrder.CW:
            indices = ensure_winding_order_triangle(v0, v1,v2, WindingOrder.CW)


        return Triangle(
            pack_vertices[indices[0]],
            pack_vertices[indices[1]],
            pack_vertices[indices[2]],
            pack_uvs[indices[0]],
            pack_uvs[indices[1]],
            pack_uvs[indices[2]]
        )

    def remove_triangle(self, index):
        """
        Remove the triangle at the specified index.
        :param index: The index of the triangle to remove.
        """
        if index < 0 or index >= self.get_triangle_count():
            raise IndexError("Triangle index out of range.")

        self.vertices = np.delete(self.vertices, slice(index * 3, index * 3 + 3), axis=0)
        self.uvs = np.delete(self.uvs, slice(index * 3, index * 3 + 3), axis=0)

    def print_triangle_info(self):
        """
        Print information about each triangle in the geometry.
        """
        for i in range(self.get_triangle_count()):
            triangle = self.get_triangle(i)
            print(f"Triangle {i}:")
            print(f"  v0: {triangle.v0}, uv0: {triangle.uv0}")
            print(f"  v1: {triangle.v1}, uv1: {triangle.uv1}")
            print(f"  v2: {triangle.v2}, uv2: {triangle.uv2}")


def trimesh_mesh_to_geometry(mesh: trimesh.Trimesh) -> Geometry:
    """
    Convert a trimesh.Trimesh object to a Geometry object.
    :param mesh: The trimesh.Trimesh object to convert.
    :return: A Geometry object containing the vertices and UV coordinates from the mesh.
    """
    vertices = []
    uv = []

    for face in mesh.faces:
        for vertex_index in face:
            vertex = mesh.vertices[vertex_index]
            vertices.append(vertex.tolist())
            if mesh.visual.uv is not None:
                uv_coord = mesh.visual.uv[vertex_index]
                uv.append(uv_coord.tolist())
            else:
                uv.append([0.0, 0.0])  # Default UV if none exist

    return Geometry(vertices=np.vstack(vertices), uvs=np.vstack(uv))