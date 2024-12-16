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
        max_velocity=1.5,
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
        self.max_velocity = max_velocity

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


def test_velocity_limits():
    """Test pour vérifier que la vitesse maximale des agents est respectée."""
    simulation = TrainStationSimulation(
        num_agents_per_team=5,
        door_position=[(-5, 5), (15, 5), (9, 8), (9, 2)],
        max_velocity=1.5,
    )
    for agent in simulation.agents:
        velocity = simulation.calculate_velocity(agent)
        assert (
            np.linalg.norm(velocity) <= simulation.max_velocity
        ), f"La vitesse de l'agent a dépassé la limite : {np.linalg.norm(velocity)}"
    print("Tous les tests de vitesse ont réussi!")


# Exécuter le test
test_velocity_limits()


def test_barrier_respect():
    """Test pour vérifier que les agents ne traversent pas la barrière en dehors du trou."""
    simulation = TrainStationSimulation(
        num_agents_per_team=5, door_position=[(5, 5)], max_velocity=1.5
    )
    for agent in simulation.agents:
        if abs(agent.position[0] - simulation.barrier_position) <= agent.radius:
            lower_bound = simulation.area_size[1] / 2 - simulation.barrier_width
            upper_bound = simulation.area_size[1] / 2 + simulation.barrier_width
            assert (
                lower_bound <= agent.position[1] <= upper_bound
            ), f"L'agent {agent} a traversé la barrière en dehors du trou."
    print("Tous les agents respectent la barrière.")


def test_agent_density():
    """Test pour vérifier que les agents maintiennent une distance minimale entre eux pendant la simulation."""
    simulation = TrainStationSimulation(
        num_agents_per_team=5, door_position=[(5, 5)], max_velocity=1.5
    )
    simulation.update_agents(dt=0.1)
    for i, agent1 in enumerate(simulation.agents):
        for j, agent2 in enumerate(simulation.agents):
            if i != j:
                distance = np.linalg.norm(agent1.position - agent2.position)
                assert distance >= (
                    agent1.radius + agent2.radius
                ), f"Collision détectée entre les agents {i} et {j} pendant la simulation."
    print("Aucune collision détectée entre les agents.")


def test_agent_density():
    """Test pour vérifier que les agents maintiennent une distance minimale entre eux pendant la simulation."""
    simulation = TrainStationSimulation(
        num_agents_per_team=5, door_position=[(5, 5)], max_velocity=1.5
    )
    simulation.update_agents(dt=0.1)
    for i, agent1 in enumerate(simulation.agents):
        for j, agent2 in enumerate(simulation.agents):
            if i != j:
                distance = np.linalg.norm(agent1.position - agent2.position)
                assert distance >= (
                    agent1.radius + agent2.radius
                ), f"Collision détectée entre les agents {i} et {j} pendant la simulation."
    print("Aucune collision détectée entre les agents.")


def test_agent_convergence():
    """Test pour vérifier que tous les agents atteignent leurs objectifs dans un temps limite."""
    simulation = TrainStationSimulation(
        num_agents_per_team=5, door_position=[(5, 5)], max_velocity=1.5
    )
    time_limit = 10
    for _ in range(int(time_limit / 0.1)):
        simulation.update_agents(dt=0.1)
    for agent in simulation.agents:
        distance_to_objective = np.linalg.norm(
            agent.position - np.array([agent.objective, 5])
        )
        assert (
            distance_to_objective <= agent.radius
        ), f"L'agent {agent} n'a pas atteint son objectif dans le temps imparti."
    print("Tous les agents ont atteint leurs objectifs.")
