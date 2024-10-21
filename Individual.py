import dataclasses
import



@dataclasses.dataclass
class Individual:
    def __init__(
        self,
        radius,
        color,
        coord,
        speed,
        id,
    ):
        self._radius = radius
        self._color = color
        self._coord = coord
        self._speed = speed
        self._id = id

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, new_radius):
        self._radius = new_radius
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name



    
    def shock(self):

    def inCollision(self):


class Indivudalpatient(indivudal)