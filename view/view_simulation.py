import pygame
import pygame_gui
import random
from multiprocessing import Process, Manager
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND
from model.animation import launch_simulation  # Pour lancer les simulations


class SimulationView:
    def __init__(self, screen):
        self.screen = screen
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Titre
        self.title_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.08))
        self.title_text = "Simulation Personnalisée"

        # Police perso
        self.label_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.035))

        # Constantes pour positionner les différents inputs

        self.vertical_gap = SCREEN_HEIGHT * 0.12  # Espace vertical entre chaque ligne
        self.top_margin = SCREEN_HEIGHT * 0.20  # Point de départ en hauteur
        self.label_height = SCREEN_HEIGHT * 0.04  # Hauteur d'un label
        self.input_height = SCREEN_HEIGHT * 0.05  # Hauteur d'un input box / slider

        # Largeur de chaque "colonne" (2 colonnes par ligne)
        self.col_width = SCREEN_WIDTH * 0.30
        # Espace horizontal entre les 2 colonnes
        self.col_spacing = SCREEN_WIDTH * 0.05

        # Calcul de la coord. d'abscisse pour la 1ère et la 2ᵉ colonne, centrées
        total_width = (2 * self.col_width) + self.col_spacing
        start_x = (SCREEN_WIDTH - total_width) / 2
        self.col1_x = start_x
        self.col2_x = start_x + self.col_width + self.col_spacing

        # Ligne 1 :

        row1_y = self.top_margin

        # Alpha :
        self.alpha_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col1_x, row1_y), (self.col_width, self.label_height)
            ),
            text="Poids Objectif (alpha) :",
            manager=self.manager,
        )
        self.alpha_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col1_x, row1_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.alpha_input.set_text("5.0")

        # Beta :
        self.beta_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col2_x, row1_y), (self.col_width, self.label_height)
            ),
            text="Poids Densité (beta) :",
            manager=self.manager,
        )
        self.beta_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col2_x, row1_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.beta_input.set_text("2.0")

        # Ligne 2 :

        row2_y = row1_y + self.vertical_gap

        # Gamma :
        self.gamma_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col1_x, row2_y), (self.col_width, self.label_height)
            ),
            text="Poids Zigzag (Gamma) :",
            manager=self.manager,
        )
        self.gamma_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col1_x, row2_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.gamma_input.set_text("0.005")  # Valeur par défaut

        # Nombre de simulations :
        self.num_simulations_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col2_x, row2_y), (self.col_width, self.label_height)
            ),
            text="Nombre de simulations :",
            manager=self.manager,
        )
        self.num_simulations_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col2_x, row2_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.num_simulations_input.set_text("1")

        # Ligne 3 :

        row3_y = row2_y + self.vertical_gap

        # Temps limite :
        self.time_limit_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col1_x, row3_y), (self.col_width, self.label_height)
            ),
            text="Temps limite (s) :",
            manager=self.manager,
        )
        self.time_limit_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col1_x, row3_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.time_limit_input.set_text("2000")

        # Nombre d'agents par équipe :
        self.num_agents_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col2_x, row3_y), (self.col_width, self.label_height)
            ),
            text="Nombre d'agents train (fixe) :",
            manager=self.manager,
        )
        self.num_agents_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (self.col2_x, row3_y + self.label_height),
                (self.col_width, self.input_height),
            ),
            manager=self.manager,
        )
        self.num_agents_input.set_text("20")

        # Ligne 4 :

        row4_y = row3_y + self.vertical_gap

        # Min Agents pour les simuls. aléatoires :
        self.min_agents_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col1_x, row4_y), (self.col_width, self.label_height)
            ),
            text="Min Agents :",
            manager=self.manager,
        )
        self.min_agents_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                (self.col1_x, row4_y + self.label_height),
                (self.col_width - 50, self.input_height),
            ),
            start_value=10,
            value_range=(1, 50),
            manager=self.manager,
        )
        self.min_agents_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col1_x + self.col_width - 50, row4_y + self.label_height),
                (50, self.input_height),
            ),
            text=str(int(self.min_agents_slider.get_current_value())),
            manager=self.manager,
        )

        # Max Agents pour les simuls. aléatoires :
        self.max_agents_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col2_x, row4_y), (self.col_width, self.label_height)
            ),
            text="Max Agents :",
            manager=self.manager,
        )
        self.max_agents_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(
                (self.col2_x, row4_y + self.label_height),
                (self.col_width - 50, self.input_height),
            ),
            start_value=40,
            value_range=(30, 55),
            manager=self.manager,
        )
        self.max_agents_value_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(
                (self.col2_x + self.col_width - 50, row4_y + self.label_height),
                (50, self.input_height),
            ),
            text=str(int(self.max_agents_slider.get_current_value())),
            manager=self.manager,
        )

        # Ligne 5 :

        row5_y = row4_y + self.vertical_gap

        # Afficher ou non les animations :

        self.show_animation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.col1_x, row5_y), (self.col_width, self.input_height)
            ),
            text="Animations : ON",
            manager=self.manager,
        )
        self.show_animation_enabled = True

        # Lancer simulation (non aléatoire) :
        self.simulation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.col2_x, row5_y), (self.col_width, self.input_height)
            ),
            text="Lancer Simulation",
            manager=self.manager,
        )

        # Ligne 6 :

        row6_y = row5_y + self.vertical_gap * 0.8

        # Retour au Menu principal :
        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.col1_x, row6_y), (self.col_width, self.input_height)
            ),
            text="Retour",
            manager=self.manager,
        )

        # Lancer simulations (aléatoires) :
        self.simulation_random_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.col2_x, row6_y), (self.col_width, self.input_height)
            ),
            text="Lancer des Simulations Aléatoires",
            manager=self.manager,
        )

        # Stocker les données partagées
        self.shared_data = {}  # Permet le partage de données entre processus

    def update(self, time_delta):
        self.screen.fill(LIGHT_BACKGROUND)

        # Afficher le titre
        title_surface = self.title_font.render(self.title_text, True, (0, 0, 0))
        title_rect = title_surface.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.1)
        )
        self.screen.blit(title_surface, title_rect)

        # Mise à jour et rendu de l'UI
        self.manager.update(time_delta)
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            # Boutons :

            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:

                # (1) "Lancer la Simulation"
                if event.ui_element == self.simulation_button:
                    alpha = float(self.alpha_input.get_text())
                    beta = float(self.beta_input.get_text())
                    gamma = float(self.gamma_input.get_text())
                    num_simulations = int(self.num_simulations_input.get_text())
                    time_limit = float(self.time_limit_input.get_text())
                    show_animation = self.show_animation_enabled

                    # Nombre d'agents (fixe)
                    num_agents = int(self.num_agents_input.get_text())

                    with Manager() as manager:
                        shared_data = manager.dict()
                        processes = []
                        for i in range(num_simulations):
                            p = Process(
                                target=launch_simulation,
                                args=(
                                    num_agents,
                                    shared_data,
                                    alpha,
                                    beta,
                                    gamma,  # On transmet gamma
                                    None,  # Si on veut par la suite choisir un fichier perso
                                    i + 1,
                                    show_animation,
                                    time_limit,
                                ),
                            )
                            processes.append(p)
                            p.start()

                        for p in processes:
                            p.join()

                        self.shared_data = dict(shared_data)

                # (2) "Lancer X simulations aléatoires"
                elif event.ui_element == self.simulation_random_button:
                    alpha = float(self.alpha_input.get_text())
                    beta = float(self.beta_input.get_text())
                    gamma = float(self.gamma_input.get_text())
                    num_simulations = int(self.num_simulations_input.get_text())
                    time_limit = float(self.time_limit_input.get_text())
                    show_animation = self.show_animation_enabled

                    # Récupération min / max depuis les sliders
                    min_agents = int(self.min_agents_slider.get_current_value())
                    max_agents = int(self.max_agents_slider.get_current_value())
                    if min_agents > max_agents:
                        min_agents, max_agents = max_agents, min_agents
                    if min_agents > max_agents:  # En cas de problème
                        min_agents, max_agents = max_agents, min_agents

                    with Manager() as manager:
                        shared_data = manager.dict()
                        processes = []
                        for i in range(num_simulations):
                            nb_rand = random.randint(min_agents, max_agents)
                            p = Process(
                                target=launch_simulation,
                                args=(
                                    nb_rand,
                                    shared_data,
                                    alpha,
                                    beta,
                                    gamma,  # On transmet gamma
                                    None,
                                    i + 1,
                                    show_animation,
                                    time_limit,
                                ),
                            )
                            processes.append(p)
                            p.start()

                        for p in processes:
                            p.join()

                        self.shared_data = dict(shared_data)

                # (3) "Retour"
                elif event.ui_element == self.back_button:
                    return "menu"

                # (4) "Animations : ON/OFF" (toggle)
                elif event.ui_element == self.show_animation_button:
                    self.show_animation_enabled = not self.show_animation_enabled
                    new_text = (
                        "Animations : ON"
                        if self.show_animation_enabled
                        else "Animations : OFF"
                    )
                    self.show_animation_button.set_text(new_text)

            # Gestion des sliders
            if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.min_agents_slider:
                    self.min_agents_value_label.set_text(
                        str(int(self.min_agents_slider.get_current_value()))
                    )
                elif event.ui_element == self.max_agents_slider:
                    self.max_agents_value_label.set_text(
                        str(int(self.max_agents_slider.get_current_value()))
                    )

        return "simulation"
