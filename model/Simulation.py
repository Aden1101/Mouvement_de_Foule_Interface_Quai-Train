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
        cooldown=5.0,
    ):
        self.position = np.array(position, dtype=float)
        self.side = side  # 1 pour ceux qui descendent, -1 pour ceux qui montent
        self.radius = np.random.uniform(0.1, 0.14)  # Taille qui varie
        self.speed = speed
        self.color = color
        self.politeness = politeness  # Paramètre de politesse (0 : impoli, 1 : poli)
        self.has_crossed = False  # Indique si l'agent a passé la porte
        self.objective = objective
        self.velocity = np.zeros(2, dtype=float)
        self.previous_velocity = np.zeros(2, dtype=float)  # Pour la pénalité Gamma
        self.cooldown = cooldown  # Temps de repositionnement
        self.next_reassign_time = 0.0

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
        door_position=[(-5, 2.5)],  # Position objectif sortant
        area_size=(4, 5),  # Taille de la zone
        max_time=20,
        barrier_width=0.35,  #  barrière un peu plus fine
        alpha_value=5.0,
        beta_value=2.0,
        max_velocity=1.5,
        gamma_zigzag=0.005,
        custom_barrier_position=3,
        cell_size=0.5,
    ):

        self.area_size = area_size
        self.cell_size = cell_size  # stocke la taille d’une cellule
        self.init_density_grid()  # on initialise la grille (voir ci-après)
        if custom_barrier_position is not None:
            self.barrier_position = custom_barrier_position
        else:
            self.barrier_position = (
                self.area_size[0] / 2
            )  # Barrière verticale au centre
        self.alpha = alpha_value  # Poids pour l'objectif
        self.beta = beta_value  # Pénalité de densité
        self.gamma_zigzag = gamma_zigzag  # Pénalité de changement de direction
        self.num_agents_per_team = num_agents_per_team
        self.door_position = np.array(door_position, dtype=float)
        self.barrier_width = barrier_width
        self.agents = self._initialize_agents()
        self.max_time = max_time  # Temps relatif pour la politesse
        self.current_time = 0  # Temps écoulé depuis le début
        self.all_blues_crossed = False
        self.max_velocity = max_velocity

    def _initialize_agents(self):
        """Initialise deux équipes d'agents tout en évitant les chevauchements initiaux."""
        agents = []
        y_min_factor = 1 / 5  # Étendu (plus bas)
        y_max_factor = 4 / 5  # Étendu (plus haut)

        # for i in range(self.num_agents_per_team):
        for i in range(2):
            # Décision 10% verts statiques, 90% bleus
            is_static = np.random.rand() < 0.15

            # Rayon tiré aléatoirement
            radius = np.random.uniform(0.1, 0.14)

            # On boucle jusqu’à trouver une position valide (sans chevauchement)
            while True:
                # 1) Intervalle X : [barrier_position + 0.14, area_size[0] - 0.14]
                x_min = self.barrier_position + 0.14
                x_max = self.area_size[0] - 0.14
                x_val = np.random.uniform(x_min, x_max)

                # 2) Intervalle Y : [0.14, area_size[1] - 0.14]
                y_min = 0.14
                y_max = self.area_size[1] - 0.14

                if is_static:
                    # (10%) : agent vert statique => Y tiré selon Beta(0.5, 0.5)
                    u = np.random.beta(0.5, 0.5)  # distribution en "U"
                    y_val = y_min + (y_max - y_min) * u
                else:
                    # (90%) : agent bleu => Y complètement uniforme
                    y_val = np.random.uniform(y_min, y_max)

                position = np.array([x_val, y_val])

                # Test de collision
                if not any(
                    np.linalg.norm(position - other.position) < (radius + other.radius)
                    for other in agents
                ):
                    # => OK, on peut sortir de la boucle
                    break

            # Une fois la position validée, on crée l’agent
            if is_static:
                spawned_agent = Agent(
                    position=position,
                    color="green",
                    side=2,  # vert statique
                    objective=position,  # ne bouge pas
                )
            else:
                spawned_agent = Agent(
                    position=position,
                    color="blue",
                    side=1,  # bleu
                    objective=self.door_position[0],  # ex : la 1ère porte
                )

            agents.append(spawned_agent)

        for i in range(int(1.5 * self.num_agents_per_team)):
            # Équipe 2 : Montent (à gauche de la porte)
            while True:
                if np.random.rand() < 0.5:
                    # Zone en haut
                    y_range = np.random.uniform(
                        self.area_size[1] / 2 + self.barrier_width + 0.14,
                        self.area_size[1] * y_max_factor,
                    )
                else:
                    # Zone en bas
                    y_range = np.random.uniform(
                        self.area_size[1] * y_min_factor,
                        self.area_size[1] / 2 - self.barrier_width - 0.14,
                    )
                x_range = np.random.uniform(
                    self.barrier_position / 3, self.barrier_position - 0.14
                )
                position = np.array([x_range, y_range])
                radius = np.random.uniform(0.1, 0.14)
                if not any(
                    np.linalg.norm(position - other.position) < (radius + other.radius)
                    for other in agents
                ):
                    break
            politeness = np.clip(
                np.random.lognormal(mean=0, sigma=0.5), 0, 1
            )  # Niveau de politesse biaisé vers 1 (poli)

            # On génère l'objectif de manière aléatoire
            margin = 0.2

            # 1) X : un uniforme sur [barrier_position + 0.2, area_size[0] - 0.2]
            x_min = self.barrier_position + margin
            x_max = self.area_size[0] - margin
            objective_x = np.random.uniform(x_min, x_max)

            # 2) Y biaisé : Beta(0.5, 0.5) => plus de chances vers 0 ou 1 qu'au centre
            # On interpole ensuite entre [margin, area_size[1] - margin]
            u = np.random.beta(0.5, 0.5)
            y_min = margin
            y_max = self.area_size[1] - margin
            objective_y = y_min + (y_max - y_min) * u

            if (
                objective_y < self.area_size[1] / 2 + 0.4
                and objective_y > self.area_size[1] / 2 - 0.4
            ):
                if objective_x < self.area_size[0] - 0.7:
                    objective_x = objective_x + 0.5

            objective = (objective_x, objective_y)

            agents.append(
                Agent(
                    position,
                    "red",
                    side=-1,
                    politeness=politeness,
                    objective=objective,
                )
            )
        print("Les agents ont tous leur positions")
        return agents

    def init_density_grid(self):
        """
        Initialise la grille de densité en fonction de self.cell_size
        """
        nx = int(np.ceil(self.area_size[0] / self.cell_size))
        ny = int(np.ceil(self.area_size[1] / self.cell_size))
        # On stocke la grille en shape (ny, nx), i.e. row=axe Y, col=axe X
        self.density_grid = np.zeros((ny, nx), dtype=float)

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
        """
        Calcule la fonction d'utilité en combinant:
        - distance_penalty
        - density_penalty
        - zigzag_penalty

        Retourne "inf" si l'agent est dans une situation impossible (collision barrière, collision agent).
        """

        # 0) Vérifier si l'agent sort ou touche le bord de area_size
        x, y = agent.position
        r = agent.radius
        # S'il dépasse un des 4 bords:
        if (x + r > self.area_size[0]) or (y - r < 0) or (y + r > self.area_size[1]):
            return float("inf")

        # 0.5) Vérifier si les agents immobiles se font sortir de la zone
        if agent.side == 2:
            if x < self.barrier_position - 0.4:
                return float("inf")

        # 1) Vérifier la collision avec la barrière en dehors du trou
        if abs(agent.position[0] - self.barrier_position) <= agent.radius:
            lower_bound = self.area_size[1] / 2 - self.barrier_width
            upper_bound = self.area_size[1] / 2 + self.barrier_width
            if not (lower_bound <= agent.position[1] <= upper_bound):
                return float("inf")

        # 2) Vérifier si l'agent chevauche un autre agent
        for other_agent in self.agents:
            if other_agent is not agent:
                dist = np.linalg.norm(agent.position - other_agent.position)
                if dist < (agent.radius + other_agent.radius):
                    return float("inf")

        # 3) Calculer séparément les trois composantes
        dist_pen = self.alpha * self.compute_distance(agent)
        dens_pen = self.compute_density_penalty(agent)
        zig_pen = self.compute_zigzag_penalty(agent)

        if agent.side == 2:
            # Densité locale
            local_density = self.get_neighborhood_density(agent.position)

            # L'agent est à son objectif, ne bouge plus en principe,
            # MAIS s’il est poli et qu'il y a trop de monde autour, il peut se pousser un peu.
            if local_density >= 3:  # Seuil
                dist_pen = dist_pen * 2
        return dist_pen + dens_pen + zig_pen

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

        norms = np.linalg.norm(all_directions, axis=1, keepdims=True)

        # Pour éviter la division par zéro, on met la vitesse à 0 si la norme est nulle.
        # Par exemple, on considère "zéro" si norme < 1e-9.
        epsilon = 1e-9
        velocities = np.zeros_like(all_directions)
        # On applique la normalisation seulement là où la norme est suffisante
        valid_indices = (norms > epsilon).reshape(-1)
        velocities[valid_indices] = (
            all_directions[valid_indices] / norms[valid_indices]
        ) * agent.speed

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

        self.update_cell()  # On met à jour la grille de densité
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
                        if agent.has_crossed:
                            # Vérifier si la densité à l’objectif est trop élevée
                            row_o, col_o = self.get_cell_indices(agent.objective)
                            if row_o is not None and col_o is not None:
                                if self.density_grid[row_o, col_o] >= 3:
                                    # Cooldown
                                    if self.current_time >= agent.next_reassign_time:
                                        new_spot = self.find_less_dense_spot(agent)
                                        agent.objective = new_spot
                                        # Maj Cooldown
                                        agent.next_reassign_time = (
                                            self.current_time + agent.cooldown
                                        )

                        if self.current_time < time_factor:
                            continue
            elif agent.side == 2:
                # Comparer densité à l'objectif et la densité de position courante
                density_current = self.get_density_at(agent.position)
                density_objective = self.get_density_at(agent.objective)
                # S'il a été "poussé" hors de son objective
                dist_obj = np.linalg.norm(agent.position - agent.objective)
                if self.current_time >= agent.next_reassign_time:
                    if dist_obj > agent.radius * 0.5:

                        if density_current < 3 and density_objective >= 3:
                            # Nouveau endroit moins dense -> on adopte ce nouveau point comme objective
                            agent.objective = agent.position.copy()
                            agent.side = -1

                        elif density_current >= 3 and density_objective >= 3:
                            agent.objective = self.find_less_dense_spot(agent)
                            agent.side = -1

                    agent.next_reassign_time = self.current_time + agent.cooldown

            distance = self.compute_distance(agent)
            # Si c'est un agent rouge et qu'il est très proche de son objectif
            if (
                agent.side == -1
                and distance < 0.14
                and agent.position[0] > self.barrier_position
            ):
                agent.side = 2

            density_penalty = self.compute_density_penalty(agent)

            if agent.side == 2 and density_penalty < 0.05 and distance < 0.15:
                continue

            velocity = self.calculate_velocity(agent)
            agent.update_position(velocity, dt)

    def compute_distance(self, agent):
        """
        Calcule la distance entre l'agent et son objectif.
        """
        # Si l'agent se fait repousser de l'autre côté alors qu'il a déjà traversé
        if (
            (agent.side == -1 and agent.position[0] < (self.barrier_position - 0.2))
            or (agent.side == 1 and agent.position[0] > (self.barrier_position + 0.2))
            and agent.has_crossed
            and (
                agent.position[1] > self.area_size[1] / 2 + 0.4
                or agent.position[1] < self.area_size[1] / 2 - 0.4
            )
        ):
            agent.has_crossed = False

        # 1) Détermination de "distance_objective"
        # On vérifie d’abord si l’agent n’a pas encore "traversé" la barrière
        if (not agent.has_crossed) and (
            (agent.side == 1 and agent.position[0] > self.barrier_position)
            or (agent.side == -1 and agent.position[0] < self.barrier_position)
        ):

            # distance horizontale
            dx = self.barrier_position - agent.position[0]

            # selon la position de y, on calcule dy :
            if agent.position[1] < self.area_size[1] / 2 - self.barrier_width / 2:
                # le point est en dessous
                dy = self.area_size[1] / 2 - self.barrier_width / 2 - agent.position[1]
            elif agent.position[1] > self.area_size[1] / 2 + self.barrier_width / 2:
                # le point est au-dessus
                dy = self.area_size[1] / 2 + self.barrier_width / 2 - agent.position[1]
            else:
                # le point est à hauteur du segment
                dy = 0

            distance_objective = np.sqrt(dx**2 + dy**2)

        else:
            # L’agent a traversé, ou a déjà traversé : on vise les objetifs
            agent.has_crossed = True
            direction_to_door = agent.objective - agent.position
            distance_objective = np.linalg.norm(direction_to_door)

        # 2) Retourne alpha * distance
        return distance_objective

    def compute_density_penalty(self, agent):
        """
        Calcule la pénalité de densité = somme( 1/(coefficient * distance + 1e-3) )
        selon side = 2 (verts), side = 1 (bleus) et side=-1 (rouges).
        """
        total_density_pen = 0
        # On scanne les autres agents
        for other_agent in self.agents:
            if other_agent is not agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                # Le 1.4 correspond a la capacité à laisser passer les autres agents des verts
                if (
                    distance < 1.3 * (agent.radius + other_agent.radius)
                    and agent.side == 2
                ):
                    total_density_pen += 1 / (2 + distance + 1e-3)

                # On teste la proximité
                if (
                    distance < 1.1 * (agent.radius + other_agent.radius)
                    and agent.side != 2
                ):
                    if other_agent.side == agent.side:
                        total_density_pen += 1 / (35 * distance + 1e-3)
                    else:
                        total_density_pen += 1 / (15 * distance + 1e-3)

        # On multiplie par beta
        return self.beta * total_density_pen

    def compute_zigzag_penalty(self, agent):
        """
        Calcule la pénalité de zigzag = gamma_zigzag * (1 - cosθ),
        où cosθ est l’angle entre velocity et previous_velocity.
        """
        zigzag_pen = 0
        if (
            np.linalg.norm(agent.previous_velocity) > 0
            and np.linalg.norm(agent.velocity) > 0
        ):
            cos_theta = np.dot(agent.velocity, agent.previous_velocity) / (
                np.linalg.norm(agent.velocity) * np.linalg.norm(agent.previous_velocity)
            )
            # Pénalité = (1 - cosθ)
            zigzag_pen = 1 - cos_theta
        return self.gamma_zigzag * zigzag_pen

    def get_cell_indices(
        self, position
    ):  # Pour savoir dans quelle cellule se trouve un agent
        x, y = position
        if x < 0 or x >= self.area_size[0] or y < 0 or y >= self.area_size[1]:
            return None, None  # hors de la zone
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def update_cell(self):
        # 1) Réinitialiser la grille à zéro
        self.density_grid.fill(0)

        # 2) Pour chaque agent, on parcourt toutes les cellules potentiellement concernées par sa bounding-box
        for ag in self.agents:
            x, y = ag.position
            r = ag.radius

            x_low = x - r
            x_high = x + r
            y_low = y - r
            y_high = y + r

            # Indices min/max en colonnes/ligne
            col_min = int(np.floor(x_low // self.cell_size))
            col_max = int(np.floor(x_high // self.cell_size))
            row_min = int(np.floor(y_low // self.cell_size))
            row_max = int(np.floor(y_high // self.cell_size))

            # On borne pour éviter d'accéder hors grille
            ny, nx = self.density_grid.shape
            col_min = max(col_min, 0)
            col_max = min(col_max, nx - 1)
            row_min = max(row_min, 0)
            row_max = min(row_max, ny - 1)

            # Parcours des cellules candidates
            for row in range(row_min, row_max + 1):
                for col in range(col_min, col_max + 1):
                    # Bounding-box de la cellule
                    xCellMin = col * self.cell_size
                    xCellMax = (col + 1) * self.cell_size
                    yCellMin = row * self.cell_size
                    yCellMax = (row + 1) * self.cell_size

                    # Test de recouvrement
                    overlap = (
                        (x_low <= xCellMax)
                        and (x_high >= xCellMin)
                        and (y_low <= yCellMax)
                        and (y_high >= yCellMin)
                    )
                    if overlap:
                        self.density_grid[row, col] += 1

    def find_less_dense_spot(self, agent, max_radius=5):
        """
        Cherche un carreau de la grille le plus proche de l'agent
        où la densité est suffisamment faible,
        Retourne un np.array([x, y]) comme nouvelle position-objectif
        Si rien trouvé, on renvoie l'objectif original.
        """
        row_agent, col_agent = self.get_cell_indices(agent.position)
        if row_agent is None or col_agent is None:
            return agent.objective  # hors zone ?

        # BFS sur les anneaux successifs
        for r in range(max_radius + 1):
            # On parcourt toutes les cellules (row, col) s.t. |row-row_agent| <= r et |col-col_agent| <= r
            rows_range = range(row_agent - r, row_agent + r + 1)
            cols_range = range(col_agent - r, col_agent + r + 1)
            for rr in rows_range:
                for cc in cols_range:
                    # On vérifie la distance en "carré" pour faire un diamond, ou en "cercle"
                    dist_manhattan = abs(rr - row_agent) + abs(cc - col_agent)
                    if dist_manhattan <= r:
                        # check si c’est dans la grille
                        if (
                            0 <= rr < self.density_grid.shape[0]
                            and 0 <= cc < self.density_grid.shape[1]
                        ):
                            # Filtrage par la densité
                            if (
                                self.density_grid[rr, cc] < 3
                            ):  # Seuil de densité acceptable
                                # Convertit (rr, cc) en (x, y)
                                x_center = (cc + 0.5) * self.cell_size
                                y_center = (rr + 0.5) * self.cell_size

                                # ⚠ Ajout du filtre "dans le train" : x_center >= barrier_position
                                if x_center >= self.barrier_position:
                                    return np.array([x_center, y_center])

        # Si pas trouvé, on garde l’objectif d’origine
        return agent.objective

    def get_density_at(self, pos):
        row, col = self.get_cell_indices(pos)
        if row is None or col is None:
            return 999999  # En dehors du terrain =
        return self.density_grid[row, col]

    def get_neighborhood_density(self, pos):
        """Calcule la densité moyenne dans la cellule de l’agent + ses voisines."""
        row, col = self.get_cell_indices(pos)
        if row is None or col is None:
            return 999999  # Hors de la grille => densité énorme

        neighbors = []
        for rr in range(row - 1, row + 2):
            for cc in range(col - 1, col + 2):
                if (
                    0 <= rr < self.density_grid.shape[0]
                    and 0 <= cc < self.density_grid.shape[1]
                ):
                    neighbors.append(self.density_grid[rr, cc])
        if len(neighbors) == 0:
            return 999999
        return np.mean(neighbors)
