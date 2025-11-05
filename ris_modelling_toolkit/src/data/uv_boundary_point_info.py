import numpy

from ris_modelling_toolkit.src.data import Axis2D

class UVBoundaryPointInfo:
    """
    The UVBoundaryPointInfo class represents info about a point in 3D where the UV coordinates cross a [0,1] boundary.
    It contains the 3D point and the associated UV axis information.
    This class is used to insert new point into a mesh when UV coordinates go outside the [0,1] range.
    """
    def __init__(self, point: numpy.ndarray[tuple[float]], uv_axis: Axis2D):
        """
        A boundary point in 3D space with associated UV axis information.
        :param point: The 3D coordinate of the boundary point.
        :param uv_axis: The UV axis (X or Y) where the boundary crossing occurs.
        """
        if len(point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        self.point = point
        self.uv_axis = uv_axis

    def __repr__(self):
        return f"BoundaryPoint(point={self.point}, uv_axis={self.uv_axis})"