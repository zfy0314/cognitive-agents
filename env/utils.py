from typing import Dict, Iterable, NamedTuple

orientation_bindings = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}


def angle_norm(theta: float) -> float:
    # normalize the angle to [0, 360)
    while theta >= 360:
        theta -= 360
    while theta < 0:
        theta += 360
    return theta


def angle_negate(theta: float) -> int:

    theta = angle_norm(theta)
    return int(180 - theta if theta < 180 else theta - 180)


def angle_abs(theta: float) -> float:

    normed = angle_norm(theta)
    return min(normed, 360 - normed)


def angle_round(theta: float) -> int:

    return min(orientation_bindings.keys(), key=lambda x: angle_abs(x - theta))


class Pos2D(NamedTuple):
    x: float
    z: float

    def __add__(self, other) -> "Pos2D":

        if isinstance(other, Pos2D):
            return Pos2D(self.x + other.x, self.z + other.z)
        elif isinstance(other, tuple) and len(other) == 2:
            return Pos2D(self.x + other[0], self.z + other[1])
        else:
            raise ValueError("Can't add {} to Pos2D".format(type(other)))


class Pos4D(NamedTuple):
    x: float
    y: float
    z: float
    theta: float

    def __add__(self, other) -> "Pos4D":

        if isinstance(other, Pos4D):
            if other.theta != self.theta or other.y != self.y:
                raise ValueError(
                    "Pos4D addition unsupported between different theta or y"
                )
            return Pos4D(self.x + other.x, self.y, self.z + other.z, self.theta)
        elif isinstance(other, Pos2D):
            return Pos4D(self.x + other.x, self.y, self.z + other.z, self.theta)
        elif isinstance(other, tuple) and len(other) == 2:
            return Pos4D(self.x + other[0], self.y, self.z + other[1], self.theta)
        else:
            raise ValueError("Can't add {} to Pos4D".format(type(other)))

    def __sub__(self, other) -> float:

        if isinstance(other, Pos4D):
            cost = (
                abs(other.x - self.x)
                + abs(other.z - self.z)
                + angle_abs(other.theta - self.theta) / 9
            )
            return cost
        else:
            raise ValueError("Can't subtract {} from Pos4D".format(type(other)))

    def __str__(self) -> str:
        return "<Pos4D: x: {:.2f}, y: {:.2f}, z: {:.2f}, theta: {:3d}>".format(
            self.x, self.y, self.z, self.theta
        )

    def snap_to_grid(self, grid: Iterable[Dict[str, float]]) -> "Pos4D":

        closest = min(
            grid, key=lambda pos: abs(pos["x"] - self.x) + abs(pos["z"] - self.z)
        )
        return Pos4D(closest["x"], self.y, closest["y"], self.theta)

    def roughly_equal(self, other: "Pos4D") -> bool:

        return self.theta == other.theta and self - other < 0.5
