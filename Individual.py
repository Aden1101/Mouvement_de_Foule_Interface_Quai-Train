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
        politeness,
    ):
        self._radius = radius
        self._color = color
        self._coord = coord
        self._speed = speed
        self._id = id
        self._politeness = politeness

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
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color):
        self._color = new_color
    
    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, new_coord):
        self._coord = new_coord

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, new_speed):
        self._speed = new_speed

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def politeness(self):
        return self._politeness

    @politeness.setter
    def coord(self, new_politeness):
        self._politeness = new_politeness
    
    def inCollision(self,list_individuals):
        for individual in list_individuals:
            if individual.id != self.id:
                if(individual.coord + individual.radius == self.coord):
                    return True
        return False       
     
    def move(self):
        "calcul gradient descente de gradient sur la fct d'utilité"
        return


class Indivudalpatient(Individual):
    "peut etre ajouter des sous classes prédéfinis avec les entrants et sortant"