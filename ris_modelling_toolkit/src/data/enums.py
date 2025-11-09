import enum


class Axis2D(enum.Enum):
    """
    Enum representing 2D axes for UV coordinates.
    0 - X axis
    1 - Y axis
    """
    X = 0
    Y = 1

class WindingOrder(enum.Enum):
    """
    The winding order of triangles.
    Enum values:
        CCW: Counter-Clockwise winding order.
        CW: Clockwise winding order.
    """
    CCW = 0
    CW = 1