import numpy as np


def ensure_winding_order_cw(indices: list[int], vertices: np.ndarray[tuple[float]]) -> list[int]:
    """
    Ensure that the winding order of triangles defined by indices and vertices is clockwise (CW).
    If a triangle is found to be counter-clockwise (CCW), its vertex order is reversed.
    :param indices: An array of shape (n, 3) where each row defines a triangle by indices into the vertices array.
    :param vertices: An array of shape (m, 3) where each row defines a vertex in 3D space.
    :return: A new array of indices with triangles ordered in clockwise winding.
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
    if normal[2] > 0:  # CCW winding
        # Swap the last two indices to make it CW
        corrected_indices = [idx0, idx2, idx1]

    return corrected_indices
