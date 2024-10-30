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


@dataclasses.dataclass
class Obstacle:
    def __init__(
        self,
        coord,
    ):
        self._coord = coord


@dataclasses.dataclass
class Wall:
    def __init__(
        self,
        coord,
        color,
    ):
        self._coord = coord
        self._color = color


@dataclasses.dataclass
class Simulation:
    def __init__(
        self,
        agent_list,
        map,
        timer,
        simulation_timer_condition,
    ):
        self._agent_list = agent_list
        self._map = map
        self._timer = timer
        self._simulation_timer_condition = simulation_timer_condition
