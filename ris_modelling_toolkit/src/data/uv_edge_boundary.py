import numpy as np

from ris_modelling_toolkit.src.data import Axis2D


class UVEdgeBoundary:
    """
        Describes where a mesh edge crosses integer UV tile boundaries.

        When UV coordinates extend outside the normalized [0, 1] range, an edge
        may cross one or more integer boundaries (e.g., 0 → 1 → 2). This class
        stores the geometric and UV information needed to later insert a split
        vertex at that crossing.

        For example, if an edge runs from U = 0.8 to U = 1.2, the crossed edge is 1
        on the U axis. The corresponding points and UVs for both edge endpoints are
        stored so the exact 3D intersection point can be computed later.
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
        Parameters
        ----------
        crossed_edges : np.ndarray
            Integer UV boundaries crossed along this edge (e.g., [1], or [-1, 0, 1]).
        index0, index1 : int
            The vertex indices of the original edge in the mesh.
        v0, v1 : np.ndarray
            The 3D coordinates of the edge endpoints.
        uv0, uv1 : np.ndarray
            The UV coordinates of the corresponding endpoints.
        axis : Axis2D
            The UV axis on which the boundary occurs (U or V).
        edge0, edge1 : float
            The UV values at the endpoints along the boundary axis.

        Notes
        -----
        - `min` and `max` store the sorted edge endpoints for convenience.
        - The actual intersection point is computed later using interpolation.
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

        return f"UVEdgeBoundary(crossed_edges={ce}, v0={v0}, v1={v1}, uv_axis={axis}, edge0={edge0}, edge1={edge1} min={self.min}, max={self.max})"
