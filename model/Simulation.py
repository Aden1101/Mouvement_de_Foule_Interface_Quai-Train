import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(
        self,
        position,
        color,
        side,
        objective,
        speed=0.8,
        politeness=1.0,
    ):
        self.position = np.array(position, dtype=float)
        self.side = side  # 1 pour ceux qui descendent, -1 pour ceux qui montent
        self.radius = np.random.uniform(0.1, 0.15)  # Taille qui varie
        self.speed = speed
        self.color = color
        self.politeness = politeness  # Paramètre de politesse (0 : impoli, 1 : poli)
        self.has_crossed = False  # Indique si l'agent a passé la porte
        self.objective = objective
        self.velocity = np.zeros(2, dtype=float)
        self.previous_velocity = np.zeros(2, dtype=float)  # Pour la pénalité Gamma

    def update_position(self, velocity, dt):
        self.previous_velocity = self.velocity
        self.velocity = velocity
        self.position += velocity * dt

    def draw(self, ax):
        circle = plt.Circle(self.position, self.radius, color=self.color)
        ax.add_artist(circle)
        return circle


class TrainStationSimulation:
    def __init__(
        self,
        num_agents_per_team,
        door_position,
        area_size=(10, 10),
        max_time=15,
        barrier_width=0.4,
        alpha_value=5.0,
        beta_value=2.0,
        max_velocity=1.5,
        gamma_zigzag=0.005,
    ):
        self.alpha = alpha_value  # Poids pour l'objectif
        self.beta = beta_value  # Pénalité de densité
        self.gamma_zigzag = gamma_zigzag  # Pénalité de changement de direction
        self.num_agents_per_team = num_agents_per_team
        self.door_position = np.array(door_position, dtype=float)
        self.area_size = area_size
        self.barrier_position = self.area_size[0] / 2  # Barrière verticale au centre
        self.barrier_width = barrier_width
        self.agents = self._initialize_agents()
        self.max_time = max_time  # Temps relatif pour la politesse
        self.current_time = 0  # Temps écoulé depuis le début
        self.all_blues_crossed = False
        self.max_velocity = max_velocity

    def _initialize_agents(self):
        """Initialise deux équipes d'agents tout en évitant les chevauchements initiaux."""
        agents = []

        if self.num_agents_per_team >= 30:
            y_min_factor = 1 / 4  # Étendu (plus bas)
            y_max_factor = 3 / 4  # Étendu (plus haut)
        else:
            y_min_factor = 1 / 3  # Normal (tiers)
            y_max_factor = 2 / 3

        for i in range(self.num_agents_per_team):
            # Équipe 1 : Descendent (à droite de la porte)
            while True:
                position = np.random.uniform(
                    [self.area_size[0] / 2 + 0.15, self.area_size[1] * y_min_factor],
                    [2 * self.area_size[0] / 3, self.area_size[1] * y_max_factor],
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
                        self.area_size[1] * y_max_factor,
                    )
                else:
                    # Zone en bas
                    y_range = np.random.uniform(
                        self.area_size[1] * y_min_factor,
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
            door_idx = np.random.randint(
                1, len(self.door_position)
            )  # Objectifs aléatoires
            agents.append(
                Agent(
                    position,
                    "red",
                    side=-1,
                    politeness=politeness,
                    objective=self.door_position[door_idx],
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

        # Vérifier si l'agent chevauche la barrière en dehors du trou
        # Vérifier si l'agent touche ou dépasse la barrière horizontalement
        if abs(agent.position[0] - self.barrier_position) <= agent.radius:
            lower_bound = self.area_size[1] / 2 - self.barrier_width
            upper_bound = self.area_size[1] / 2 + self.barrier_width

            if not (lower_bound <= agent.position[1] <= upper_bound):
                return float("inf")

        # Vérifier si l'agent chevauche un autre agent

        distance_objective = 0

        for other_agent in self.agents:
            if other_agent is not agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < (agent.radius + other_agent.radius):
                    return float("inf")

        # Reset l'objective si l'agent se fait repousser
        if (
            agent.side == 1
            and agent.has_crossed
            and agent.position[0] > self.barrier_position + 0.25
        ):
            agent.has_crossed = False

        if (
            agent.side == -1
            and agent.has_crossed
            and agent.position[0] < self.barrier_position - 0.25
        ):
            agent.has_crossed = False

        if (
            (not agent.has_crossed)
            and (agent.side == 1 and agent.position[0] > (self.barrier_position))
        ) or (
            (not agent.has_crossed)
            and (agent.side == -1 and agent.position[0] < (self.barrier_position))
        ):
            direction_to_door = [
                self.area_size[0] / 2,
                self.area_size[1] / 2,
            ] - agent.position
            direction_to_door = np.linalg.norm(direction_to_door)
            distance_objective = direction_to_door
        else:

            agent.has_crossed = True
            direction_to_door = agent.objective - agent.position
            distance_to_door = np.linalg.norm(direction_to_door)
            distance_objective = distance_to_door

        density_penalty = 0
        for other_agent in self.agents:
            if other_agent is not agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < 1.1 * (agent.radius + other_agent.radius):
                    if other_agent.side == agent.side:
                        density_penalty += 1 / (35 * distance + 1e-3)
                    else:
                        density_penalty += 1 / (15 * distance + 1e-3)

        # Pénalité de zigzag
        zigzag_penalty = 0
        if (
            np.linalg.norm(agent.previous_velocity) > 0
            and np.linalg.norm(agent.velocity) > 0
        ):
            cos_theta = np.dot(agent.velocity, agent.previous_velocity) / (
                np.linalg.norm(agent.velocity) * np.linalg.norm(agent.previous_velocity)
            )
            zigzag_penalty = (
                1 - cos_theta
            )  # Écart entre l'ancienne direction et la nouvelle

        return (
            self.alpha * distance_objective
            + self.beta * density_penalty
            + self.gamma_zigzag * zigzag_penalty
        )

    def _create_forced_directions(self, agent, n=5):
        """
        Crée n directions orientées vers l'objectif de l'agent,
        avec un léger bruit autour de la direction idéale.
        """
        if not agent.has_crossed:
            # 1) S'il n'a pas encore traversé la barrière
            #    On vise le centre du trou (ou barrière_position)
            target = np.array([self.barrier_position, self.area_size[1] / 2])
        else:
            # 2) Sinon, on vise la position objective (i.e. la porte)
            target = agent.objective

        direction = target - agent.position
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            # Cas limite: on est déjà sur la cible -> direction nulle
            return np.zeros((n, 2))

        direction /= dist  # on normalise

        forced = []
        for _ in range(n):
            # On ajoute un léger "bruit" aléatoire
            noise = np.random.uniform(-0.1, 0.1, size=2)
            dir_noisy = direction + noise
            norm = np.linalg.norm(dir_noisy)
            if norm > 1e-9:
                dir_noisy /= norm
            forced.append(dir_noisy)
        forced.append(direction)

        return np.array(forced)

    def calculate_velocity(self, agent):
        """
        Algorithme testant différentes directions de vitesse et choisissant
        celle qui minimise la fonction d'utilité.
        """

        # 1) 25 directions aléatoires
        random_directions = np.random.uniform(-0.5, 0.5, size=(25, 2))

        # 2) 6 directions forcées autour de l'objectif
        forced_directions = self._create_forced_directions(agent, n=5)

        # On concatène
        all_directions = np.vstack([random_directions, forced_directions])

        # Normalisation + mise à l'échelle avec la vitesse de l'agent
        velocities = (
            all_directions
            / np.linalg.norm(all_directions, axis=1, keepdims=True)
            * agent.speed
        )

        best_velocity = velocities[0]
        best_utility = float("inf")

        # On stocke la position d'origine pour ne pas la "corrompre"
        original_position = agent.position.copy()
        original_velocity = agent.velocity.copy()
        original_prev_velocity = agent.previous_velocity.copy()

        for velocity in velocities:
            # On applique virtuellement le déplacement
            agent.position = original_position + velocity * 0.1
            # On simule ce que serait agent.velocity / agent.previous_velocity
            agent.previous_velocity = original_velocity
            agent.velocity = velocity

            # Calcul d'utilité
            utility = self.calculate_utility(agent)

            # Restauration
            agent.position = original_position
            agent.velocity = original_velocity
            agent.previous_velocity = original_prev_velocity

            if utility < best_utility:
                best_utility = utility
                best_velocity = velocity

        # Si "inf" => on ne bouge pas
        if best_utility == float("inf"):
            best_velocity = np.zeros(2)

        return best_velocity

    def update_agents(self, dt=0.1):
        """Met à jour les positions de tous les agents."""
        self.current_time += dt
        self.are_all_blues_crossed()
        shuffled_list = list(self.agents)
        np.random.shuffle(
            shuffled_list
        )  # Pour éviter qu'un agent ait toujours la priorité du mouvement
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
