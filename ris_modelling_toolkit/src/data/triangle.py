import numpy as np

from ris_modelling_toolkit.src.data.line import LineSegment
from ris_modelling_toolkit.src.data.plane import Plane
from ris_modelling_toolkit.src.util import lerp


def _intersect(p1, p2, d1, d2):
    """
    Helper function to compute intersection point between edge and plane.
    :param p1: The first point of the edge.
    :param p2: The second point of the edge.
    :param d1: The distance from the plane to the first point.
    :param d2: The distance from the plane to the second point.
    :return: The intersection point as a numpy array.
    """
    t = d1 / (d1 - d2)
    return p1 + t * (p2 - p1)


class Triangle:
    """
    A class to represent a triangle in 2D space with texture coordinates.
    """

    def __init__(self,
                 v0: np.ndarray[list[float]],
                 v1: np.ndarray[list[float]],
                 v2: np.ndarray[list[float]],
                 uv0: np.ndarray[list[float]],
                 uv1: np.ndarray[list[float]],
                 uv2: np.ndarray[list[float]]):
        """
        Initialize the Triangle with vertices and UV coordinates.
        :param v0: A tuple (x, y) representing the first vertex of the triangle.
        :param v1: A tuple (x, y) representing the second vertex of the triangle.
        :param v2: A tuple (x, y) representing the third vertex of the triangle.
        :param uv0: A tuple (u, v) representing the texture coordinates for vertex1.
        :param uv1: A tuple (u, v) representing the texture coordinates for vertex2.
        :param uv2: A tuple (u, v) representing the texture coordinates for vertex3.
        """

        self._normal: None | np.ndarray[tuple[float]] = None
        if len(v0) != 3 or len(v1) != 3 or len(v2) != 3:
            raise ValueError("Vertices must be 3D coordinates.")

        if len(uv0) != 2 or len(uv1) != 2 or len(uv2) != 2:
            raise ValueError("UV coordinates must be 2D coordinates.")

        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.uv0 = uv0
        self.uv1 = uv1
        self.uv2 = uv2
        self.line_segment_a = LineSegment(self.v0, self.v1)
        self.line_segment_b = LineSegment(self.v1, self.v2)
        self.line_segment_c = LineSegment(self.v2, self.v0)

    def normal(self) -> np.ndarray[tuple[float]]:
        """
        Return the normal vector of the triangle.

        :return: A numpy array representing the normal vector of the triangle.
        """

        if self._normal is not None:
            return self._normal

        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        normal = np.cross(edge1, edge2)
        norm_length = np.linalg.norm(normal)
        if norm_length == 0:
            raise ValueError("Degenerate triangle with zero area has no valid normal.")
        self._normal = normal / norm_length
        return self._normal

    def point_in_triangle(
            self,
            point: np.ndarray[list[float]],
            tol: float = 1e-8) -> bool:
        """
        Check if a 3D point lies inside of this triangle.

        :param point: np.ndarray of shape (3,), the 3D point to test.
        :param tol: Tolerance for numerical precision.
        :return: True if the point is inside the triangle (or on its edge), False otherwise.
        """
        v0 = self.v0
        v1 = self.v1
        v2 = self.v2

        # Vectors from v0
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0

        # Compute dot products
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)

        # Compute barycentric coordinates
        denom = d00 * d11 - d01 * d01
        if np.abs(denom) < tol:
            return False  # Degenerate triangle

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        # Point is inside triangle if all barycentric cords are between 0 and 1
        return (-tol <= u <= 1 + tol) and (-tol <= v <= 1 + tol) and (-tol <= w <= 1 + tol)

    def point_on_triangle_edge(self, point: np.ndarray[list[float]], tol: float = 1e-8) -> list[bool | LineSegment] | \
                                                                                           list[bool | None]:
        """
        Check if a 3D point lies on the edge of this triangle.

        :param point: np.ndarray of shape (3,), the 3D point to test.
        :param tol: Tolerance for numerical precision.
        :return: Tuple (is_on_edge: bool, edge: LineSegment|None)
                 where is_on_edge is True if the point is on an edge,
                 and edge is the LineSegment of that edge or None if not on any edge.
        """

        if len(point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        if self.line_segment_a.is_point_on_segment(point, tol):
            return [True, self.line_segment_a]
        elif self.line_segment_b.is_point_on_segment(point, tol):
            return [True, self.line_segment_b]
        elif self.line_segment_c.is_point_on_segment(point, tol):
            return [True, self.line_segment_c]
        else:
            return [False, None]

    def copy(self) :
        """
        Create a copy of this triangle.
        :return: The copied Triangle object.
        """
        return Triangle(
            self.v0.copy(),
            self.v1.copy(),
            self.v2.copy(),
            self.uv0.copy(),
            self.uv1.copy(),
            self.uv2.copy()
        )

    def clip_by_plane(self, plane: Plane) -> list["Triangle"]:
        """
        Clip this triangle by a plane.
        :param plane: The Plane to clip against.
        :return: The resulting triangles after clipping (0, 1, or 2 triangles).
        """

        plane_n = plane.normal
        plane_d = plane.distance_from_origin()

        pts = [self.v0, self.v1, self.v2]
        uvs  = [self.uv0, self.uv1, self.uv2]
        d = [plane_n.dot(v) - plane_d for v in pts]

        above = [i for i in range(3) if d[i] >= 0]
        below = [i for i in range(3) if d[i] < 0]

        # Fully clipped
        if len(above) == 0:
            return []

        # Fully kept
        if len(above) == 3:
            return [self.copy()]

        # Helper intersection
        def I(i, j):
            return _intersect(pts[i], pts[j], d[i], d[j])

        out = []
        out_uvs = []

        # One vertex above → one output triangle
        if len(above) == 1:
            a = above[0]
            b = below[0]
            c = below[1]
            Pab = I(a, b)
            Pac = I(a, c)
            UVab = lerp(uvs[a], uvs[b], d[a] / (d[a] - d[b]))
            UVac = lerp(uvs[a], uvs[c], d[a] / (d[a] - d[c]))
            out.append((pts[a], Pab, Pac))
            out.append((pts[b], Pac, Pac))
            out_uvs.append((uvs[a], UVab, UVac))
            out_uvs.append((uvs[b], UVac, UVab))

        # Two vertices above → two output triangles
        else:
            a = above[0]
            b = above[1]
            c = below[0]
            Pac = I(a, c)
            Pbc = I(b, c)
            UVab = lerp(uvs[a], uvs[c], d[a] / (d[a] - d[c]))
            UVbc = lerp(uvs[b], uvs[c], d[b] / (d[b] - d[c]))
            out.append((pts[a], pts[b], Pbc))
            out.append((pts[a], Pbc, Pac))
            out_uvs.append((uvs[a], uvs[b], UVbc))
            out_uvs.append((uvs[a], UVbc, UVab))

        triangles = []
        for idx, triangle_pts in enumerate(out):
            triangles.append(
                Triangle(
                    triangle_pts[0],
                    triangle_pts[1],
                    triangle_pts[2],
                    out_uvs[idx][0],
                    out_uvs[idx][1],
                    out_uvs[idx][2]
                )
            )

        return triangles

    def is_valid_triangle(self, tol: float = 1e-8) -> bool:
        """
        Check if this triangle is valid (not degenerate).
        :param tol: Tolerance for numerical precision.
        :return: True if the triangle is valid, False otherwise.
        """

        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        cross_product = np.cross(edge1, edge2)
        area = np.linalg.norm(cross_product) / 2.0
        return area > tol

    def equal(self, other: "Triangle") -> bool:
        """
        Check if this triangle is equal to another triangle.
        :param other: The other Triangle to compare with.
        :return: True if the triangles are equal, False otherwise.
        """
        return (np.array_equal(self.v0, other.v0) and
                np.array_equal(self.v1, other.v1) and
                np.array_equal(self.v2, other.v2) and
                np.array_equal(self.uv0, other.uv0) and
                np.array_equal(self.uv1, other.uv1) and
                np.array_equal(self.uv2, other.uv2))