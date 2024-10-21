import dataclasses


@dataclasses.dataclass
class Coord:
    def __init__(
        self,
        x,
        y,
    ):
        self._x = x
        self._y = y
