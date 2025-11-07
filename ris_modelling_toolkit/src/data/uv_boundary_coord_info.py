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

    def __init__(self,
                 crossed_edges: np.ndarray,
                 index0: int, index1: int,
                 v0: np.ndarray[list[float]],
                 v1: np.ndarray[list[float]],
                 uv0: np.ndarray[list[float]],
                 uv1: np.ndarray[list[float]],
                 axis: Axis2D,
                 edge0: float, edge1: float):
        """
        A boundary in UV coordinate space where UVs cross specified edges (e.g., 0 or 1).
        :param crossed_edges: Edges that were crossed in UV coordinate space.
        :param face0: The first vertex index of the face.
        :param face1: The second vertex index of the face.
        :param v0 : The first 3D vertex of the edge. Reference to the original edge.
        :param v1 : The second 3D vertex of the edge. Reference to the original edge.
        :param uv0 : The first vertex UV coordinates. Reference to the original edge.
        :param uv1 : The second vertex UV coordinates. Reference to the original edge.
        :param axis: The UV axis (X or Y) for which the boundary crossing occurs.
        :param edge0: The first edge value.
        :param edge1: The second edge value.
        """

        if len(v0) != 3 or len(v1) != 3:
            raise ValueError("v0 and v1 must be 3D coordinates.")

        self.crossed_edges = crossed_edges
        self.index0 = index0
        self.index1 = index1
        self.v0 = v0
        self.v1 = v1
        self.uv0 = uv0
        self.uv1 = uv1
        self.uv_axis = axis
        self.edge0 = edge0
        self.edge1 = edge1
        self.min = min(edge0, edge1)
        self.max = max(edge0, edge1)

    def __repr__(self):
        ce = ", ".join([f"{e:.3f}" for e in self.crossed_edges])
        v0 = f"({self.v0[0]:.3f}, {self.v0[1]:.3f}, {self.v0[2]:.3f})"
        v1 = f"({self.v1[0]:.3f}, {self.v1[1]:.3f}, {self.v1[2]:.3f})"
        axis = self.uv_axis
        edge0 = self.edge0
        edge1 = self.edge1

        return f"UVBoundaryCoordInfo(crossed_edges={ce}, v0={v0}, v1={v1}, uv_axis={axis}, edge0={edge0}, edge1={edge1} min={self.min}, max={self.max})"
