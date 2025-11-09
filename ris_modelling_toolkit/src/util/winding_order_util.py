import numpy as np

from ris_modelling_toolkit.src.data.enums import WindingOrder

def ensure_winding_order_triangle(
        v0: np.ndarray[tuple[float]],
        v1: np.ndarray[tuple[float]],
        v2: np.ndarray[tuple[float]],
        winding_order = WindingOrder.CCW) -> list[int]:
    """
    Ensure that the winding order of a triangle defined by vertices v0, v1, v2 is counter-clockwise (CCW).
    If the triangle is found to be clockwise (CW), its vertex order is reversed.
    :param v0: The first vertex of the triangle.
    :param v1: The second vertex of the triangle.
    :param v2: The third vertex of the triangle.
    :param winding_order: The desired winding order of the triangle.
    :return: A list of vertex indices with the triangle ordered in counter-clockwise winding.
    """
    indices = [0, 1, 2]
    return ensure_winding_order(indices, np.array([v0, v1, v2]), winding_order)

def ensure_winding_order(indices: list[int], vertices: np.ndarray[tuple[float]], winding_order = WindingOrder.CCW) -> list[int]:
    """
    Ensure that the winding order of triangles defined by indices and vertices is counter-clockwise (CCW).
    If a triangle is found to be clockwise (CW), its vertex order is reversed.
    :param indices: An array of shape (n, 3) where each row defines a triangle by indices into the vertices array.
    :param vertices: An array of shape (m, 3) where each row defines a vertex in 3D space.
    :param winding_order: The winding order of triangles defined by indices and vertices.
    :return: A new array of indices with triangles ordered in counter-clockwise winding.
    """
    if len(indices) != 3:
        raise ValueError("Indices must define a single triangle with 3 vertices.")

    corrected_indices = indices

    idx0, idx1, idx2 = indices
    v0 = vertices[idx0]
    v1 = vertices[idx1]
    v2 = vertices[idx2]

    # Compute the normal using the cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)

    # Assuming a right-handed coordinate system, check the Z component of the normal
    if winding_order == WindingOrder.CCW and normal[2] < 0:  # CW winding
        # Swap the last two indices to make it CCW
        corrected_indices = [idx0, idx2, idx1]
    elif winding_order == WindingOrder.CW and normal[2] > 0:
        # Swap the last two indices to make it CW
        corrected_indices = [idx0, idx2, idx1]

    return corrected_indices

def ensure_winding_order_ccw(indices: list[int], vertices: np.ndarray[tuple[float]]) -> list[int]:
    """
    Ensure that the winding order of triangles defined by indices and vertices is counter-clockwise (CCW).
    If a triangle is found to be clockwise (CW), its vertex order is reversed.
    :param indices: An array of shape (n, 3) where each row defines a triangle by indices into the vertices array.
    :param vertices: An array of shape (m, 3) where each row defines a vertex in 3D space.
    :return: A new array of indices with triangles ordered in counter-clockwise winding.
    """
    return ensure_winding_order(indices, vertices, WindingOrder.CCW)

def ensure_winding_order_cw(indices: list[int], vertices: np.ndarray[tuple[float]]) -> list[int]:
    """
    Ensure that the winding order of triangles defined by indices and vertices is clockwise (CW).
    If a triangle is found to be counter-clockwise (CCW), its vertex order is reversed.
    :param indices: An array of shape (n, 3) where each row defines a triangle by indices into the vertices array.
    :param vertices: An array of shape (m, 3) where each row defines a vertex in 3D space.
    :return: A new array of indices with triangles ordered in clockwise winding.
    """
    return ensure_winding_order(indices, vertices, WindingOrder.CW)

