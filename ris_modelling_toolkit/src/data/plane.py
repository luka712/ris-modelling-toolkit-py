import numpy as np


class Plane:
    def __init__(self, point, normal):
        """
        Initialize the Plane with a point and a normal vector.
        :param point: The point on the plane (3D coordinates).
        :param normal: The normal vector of the plane (3D vector).
        """

        if len(point) != 3:
            raise ValueError("Point must be a 3D coordinate.")

        if len(normal) != 3:
            raise ValueError("Normal must be a 3D vector.")

        self.point = point  # A point on the plane (3D coordinates)
        self.normal = normal  # The normal vector of the plane (3D vector)
        self._distance_from_origin = np.dot(self.normal, self.point)

    def distance_from_origin(self) -> float:
        """
        Returns the distance from the origin to the plane along the normal vector.
        """
        return self._distance_from_origin

    def equation(self) -> tuple[float, float, float, float]:
        """
        Returns the coefficients (A, B, C, D) of the plane equation Ax + By + Cz + D = 0
        """
        a,b,c = self.normal
        d = - (a * self.point[0] + b * self.point[1] + c * self.point[2])
        return a,b,c,d

    def perpendicular_plane_from_vector(self,
                                        vector: np.ndarray[list[float]]) -> "Plane":
        """
        Create a new Plane that is perpendicular to this plane and passes through the given vector point.
        :param vector: A point (3D coordinates) through which the new plane will pass.
        :return: A new Plane object.
        """

        if len(vector) != 3:
            raise ValueError("Vector must be a 3D coordinate.")

        return Plane(point=vector, normal=self.normal)