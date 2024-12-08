import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SimulationManager:
    def __init__(self, num_agents, barrier_width, collision_distance):
        self.num_agents = num_agents
        self.barrier_width = barrier_width
        self.collision_distance = collision_distance
        self.agents_pos, self.agents_side = self.init_agents()
        self.all_blue_crossed = False

    def init_agents(self):
        agents_pos = np.zeros((self.num_agents, 2))
        agents_side = np.zeros(self.num_agents)
        # Initialize agents here...
        return agents_pos, agents_side

    def update(self):
        # Update agent positions
        pass


class Agent:
    def __init__(
        self,
        position,
        color,
        side,
        objective,
        speed=1.0,
        politeness=1.0,
    ):
        self.position = np.array(position, dtype=float)
        self.side = side  # 1 pour ceux qui descendent, -1 pour ceux qui montent
        self.radius = np.random.uniform(0.1, 0.15)
        self.speed = speed
        self.color = color
        self.politeness = politeness  # Paramètre de politesse (0 : impoli, 1 : poli)
        self.has_crossed = False
        self.objective = objective

    def update_position(self, velocity, dt):
        self.position += velocity * dt

    def draw(self, ax):
        """Dessine un cercle représentant l'agent sur les axes donnés."""
        circle = plt.Circle(self.position, self.radius, color=self.color)
        ax.add_artist(circle)
        return circle


class TrainStationSimulation:
    def __init__(
        self,
        num_agents_per_team,
        door_position,
        area_size=(10, 10),
        max_time=10,
        barrier_width=0.4,
        alpha_value=3.0,
        beta_value=3.0,
    ):
        self.alpha = alpha_value
        self.beta = beta_value
        self.num_agents_per_team = num_agents_per_team
        self.door_position = np.array(door_position, dtype=float)
        self.area_size = area_size
        self.barrier_position = self.area_size[0] / 2  # Barrière verticale au centre
        self.trou_center = np.array([self.barrier_position, self.area_size[1] / 2])
        self.barrier_width = barrier_width
        self.agents = self._initialize_agents()
        self.max_time = max_time  # Temps total de simulation
        self.current_time = 0  # Temps écoulé
        self.all_blues_crossed = False

    def _initialize_agents(self):
        """Initialise deux équipes d'agents tout en évitant les chevauchements initiaux."""
        agents = []
        for i in range(self.num_agents_per_team):
            # Équipe 1 : Descendent (à droite de la porte)
            while True:
                position = np.random.uniform(
                    [self.area_size[0] / 2 + 0.15, self.area_size[1] / 3],
                    [2 * self.area_size[0] / 3, 2 * self.area_size[1] / 3],
                )
                radius = np.random.uniform(0.1, 0.15)
                if not any(
                    np.linalg.norm(position - other.position) < (radius + other.radius)
                    for other in agents
                ):
                    break
            agents.append(
                Agent(position, "blue", side=1, objective=self.door_position[0])
            )

            # Équipe 2 : Montent (à gauche de la porte)
            while True:
                if np.random.rand() < 0.5:
                    # Zone en haut
                    y_range = np.random.uniform(
                        self.area_size[1] / 2 + self.barrier_width + 0.15,
                        2 * self.area_size[1] / 3,
                    )
                else:
                    # Zone en bas
                    y_range = np.random.uniform(
                        self.area_size[1] / 3,
                        self.area_size[1] / 2 - self.barrier_width - 0.15,
                    )
                x_range = np.random.uniform(
                    self.area_size[0] / 3, self.area_size[0] / 2 - 0.15
                )
                position = np.array([x_range, y_range])
                radius = np.random.uniform(0.1, 0.15)
                if not any(
                    np.linalg.norm(position - other.position) < (radius + other.radius)
                    for other in agents
                ):
                    break
            politeness = np.clip(
                np.random.lognormal(mean=0, sigma=0.5), 0, 1
            )  # Niveau de politesse biaisé vers 1 (poli)
            i = np.random.randint(1, len(self.door_position))
            agents.append(
                Agent(
                    position,
                    "red",
                    side=-1,
                    politeness=politeness,
                    objective=self.door_position[i],
                )
            )

        return agents

    def are_all_blues_crossed(self):
        """Vérifie si tous les bleus ont passé la barrière."""
        for agent in self.agents:
            if agent.side == 1 and agent.position[0] > (self.barrier_position - 0.3):
                return False
        self.all_blues_crossed = True
        return True

    def are_all_reds_crossed(self):
        """Vérifie si tous les rouges ont passé la barrière."""
        for agent in self.agents:
            if agent.side == -1 and agent.position[0] < self.barrier_position:
                return False
        return True

    def calculate_utility(self, agent):
        """Calcule la fonction d'utilité pour un agent."""
        utility = 0

        # Vérifier si l'agent chevauche la barrière en dehors du trou
        # Vérifier si l'agent touche ou dépasse la barrière horizontalement
        if abs(agent.position[0] - self.barrier_position) <= agent.radius:
            lower_bound = self.area_size[1] / 2 - self.barrier_width
            upper_bound = self.area_size[1] / 2 + self.barrier_width

            if not (lower_bound <= agent.position[1] <= upper_bound):
                return float("inf")

        # Vérifier si l'agent chevauche un autre agent
        for other_agent in self.agents:
            if other_agent is not agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < (agent.radius + other_agent.radius):
                    return float("inf")

        if (
            (not agent.has_crossed)
            and (agent.side == 1 and agent.position[0] > (self.barrier_position))
        ) or (
            (not agent.has_crossed)
            and (agent.side == -1 and agent.position[0] < (self.barrier_position))
        ):
            direction_to_trou = [
                self.area_size[0] / 2,
                self.area_size[1] / 2,
            ] - agent.position
            distance_to_trou = np.linalg.norm(direction_to_trou)
            utility += distance_to_trou
        else:
            agent.has_crossed = True
            direction_to_door = agent.objective - agent.position
            distance_to_door = np.linalg.norm(direction_to_door)
            utility += distance_to_door

        density_penalty = 0
        for other_agent in self.agents:
            if other_agent is not agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < 2 * agent.radius:
                    density_penalty += 1 / (distance + 1e-3)

        alpha = 3.0
        beta = 5

        return alpha * utility + beta * density_penalty

    def calculate_velocity(self, agent):
        """Calcule la meilleure direction pour minimiser la fonction d'utilité."""
        directions = np.random.uniform(-0.5, 0.5, size=(30, 2))
        velocities = (
            directions / np.linalg.norm(directions, axis=1, keepdims=True) * agent.speed
        )
        best_velocity = velocities[0]
        best_utility = float("inf")

        for velocity in velocities:
            new_position = agent.position + velocity * 0.1
            agent.position = new_position
            utility = self.calculate_utility(agent)
            agent.position -= velocity * 0.1
            if utility < best_utility:
                best_utility = utility
                best_velocity = velocity

        if best_utility == float("inf"):
            best_velocity = 0

        return best_velocity

    def update_agents(self, dt=0.1):
        """Met à jour les positions de tous les agents."""
        self.current_time += dt
        self.are_all_blues_crossed()
        shuffled_list = list(self.agents)
        np.random.shuffle(shuffled_list)
        for agent in shuffled_list:
            if agent.side == -1:
                if not self.all_blues_crossed:
                    if agent.politeness == 1:
                        continue
                    elif agent.politeness > 0:
                        time_factor = self.max_time / 2 * (1 - agent.politeness)
                        if self.current_time < time_factor:
                            continue

            velocity = self.calculate_velocity(agent)
            agent.update_position(velocity, dt)
