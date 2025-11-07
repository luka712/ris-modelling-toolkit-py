import numpy

from ris_modelling_toolkit.src.data import Axis2D

class UVBoundaryPointInfo:
    """
    The UVBoundaryPointInfo class represents info about a point in 3D where the UV coordinates cross a [0,1] boundary.
    It contains the 3D point and the associated UV axis information.
    This class is used to insert new point into a mesh when UV coordinates go outside the [0,1] range.
    Holds information about edge0 and edge1 for reference to the original UV coordinates.
    Value of edges is either U or V coordinate depending on the UV axis.
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
