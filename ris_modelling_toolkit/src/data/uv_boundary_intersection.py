import numpy

from ris_modelling_toolkit.src.data import Axis2D

class UVBoundaryIntersection:
    """
        Represents a newly created vertex where a mesh edge crosses an integer UV tile boundary.

        When UV coordinates extend outside the normalized [0, 1] range, edges may cross
        boundaries at integer UV values (e.g., U = 1, U = 2, U = -1). To correctly tile
        and unwrap the mesh, these edges must be split at the exact intersection point in
        both 3D space and UV space.

        This class stores:
          - The 3D intersection point on the mesh edge.
          - The original edge endpoints and indices.
          - Which UV axis (U or V) was crossed.
          - The specific UV boundary value (e.g., U = 1, V = -1).
          - The interpolated coordinate on the *other* UV axis at the split point.

        The information is used later during mesh reconstruction to insert the new vertex
        and update UVs consistently across splits.
        """
    def __init__(self,
                 point: numpy.ndarray[list[float]],
                 index0: int, index1: int,
                 v0: numpy.ndarray[list[float]],
                 v1: numpy.ndarray[list[float]],
                 crossed_edge: float,
                 edge0: float, edge1: float,
                 uv_axis: Axis2D,
                 other_coord_interpolated: float
                 ):
        """
        A boundary point in 3D space with associated UV axis information.
        :param point: The 3D coordinate of the boundary point.
        :param index0: The first vertex index of the face.
        :param index1: The second vertex index of the face.
        :param v0 : The first 3D vertex of the edge. Reference to the original edge.
        :param v1 : The second 3D vertex of the edge. Reference to the original edge.
        :param crossed_edge: The crossed edge in UV coordinate space.
        :param edge0: The edge of the boundary point.
        :param edge1: The other edge of the boundary point.
        :param uv_axis: The UV axis (X or Y) where the boundary crossing occurs.
        :param other_coord_interpolated: The interpolated coordinate on the other UV axis.
        """
        if len(point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        if len(v0) != 3 or len(v1) != 3:
            raise ValueError("v0 and v1 must be 3D coordinates.")

        self.point = point
        self.index0 = index0
        self.index1 = index1
        self.v0 = v0
        self.v1 = v1
        self.crossed_edge = crossed_edge
        self.edge0 = edge0
        self.edge1 = edge1
        self.uv_axis = uv_axis
        self.other_coord_interpolated = other_coord_interpolated

    def __repr__(self):
        return f"BoundaryPoint(point={self.point}, edge0={self.edge0}, edge1={self.edge1}, uv_axis={self.uv_axis}, other_coord_interpolated={self.other_coord_interpolated})"

    def get_point(self) -> numpy.ndarray[list[float]]:
        """
        Get the 3D coordinate of the boundary point.
        :return: The 3D coordinate as a numpy array.
        """
        return self.point