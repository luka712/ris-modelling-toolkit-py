import numpy as np

from ris_modelling_toolkit.src.data import Axis2D

class UVBoundaryCoordInfo:
    """
    The UVBoundaryCoordInfo class represents a class that holds boundary information in UV coordinate space where UVs cross integer edges.
    It contains the crossed edges, the UV axis information, and the min/max edge values.
    For example, if UVs cross from 0.8 to 1.2 on the U axis, this class would hold 1 as the crossed edge,
    Axis2D.X as the axis, and 0.8 and 1.2 as the min and max values.
    This class is used to track UV boundaries for mesh processing when UV coordinates go outside the [0,1] range.
    """

    def __init__(self, crossed_edges: np.ndarray, axis: Axis2D, edge0: float, edge1: float):
        """
        A boundary in UV coordinate space where UVs cross specified edges (e.g., 0 or 1).
        :param crossed_edges: Edges that were crossed in UV coordinate space.
        :param axis: The UV axis (X or Y) where the boundary crossing occurs.
        :param edge0: The first edge value.
        :param edge1: The second edge value.
        """

        self.crossed_edges = crossed_edges
        self.uv_axis = axis
        self.min = min(edge0, edge1)
        self.max = max(edge0, edge1)

    def __repr__(self):
        return f"UVBoundaryCoordInfo(crossed_edges={self.crossed_edges}, uv_axis={self.uv_axis}, min={self.min}, max={self.max})"