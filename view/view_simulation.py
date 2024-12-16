import pygame
import pygame_gui
from multiprocessing import Process, Manager
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND
from model.animation import launch_simulation


class SimulationView:
    def __init__(self, screen, simulation_manager):
        self.screen = screen
        self.simulation_manager = simulation_manager
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Titre
        self.title_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.08))
        self.title_text = "Simulation Personnalisée"

        # Police pour les étiquettes
        self.label_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.04))

        # Dimensions et position centrée pour les champs et les boutons
        field_width = SCREEN_WIDTH * 0.4
        field_height = SCREEN_HEIGHT * 0.05
        button_height = SCREEN_HEIGHT * 0.07
        x_center = (SCREEN_WIDTH - field_width) / 2

        # Champs de saisie pour les paramètres
        self.num_agents_label = "Nombre d'agents :"
        self.num_agents_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.25), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.num_agents_input.set_text("20")

        self.alpha_label = "Poids Objectif (alpha) :"
        self.alpha_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.35), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.alpha_input.set_text("3.0")

        self.beta_label = "Poids Densité (beta) :"
        self.beta_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.45), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.beta_input.set_text("3.0")

        self.num_simulations_label = "Nombre de simulations :"
        self.num_simulations_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.55), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.num_simulations_input.set_text("1")

        self.time_limit_label = "Temps limite (s) :"
        self.time_limit_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.65), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.time_limit_input.set_text("40")

        # Bouton toggle pour afficher ou non les animations
        self.show_animation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.75), (field_width, button_height)
            ),
            text="Animations : ON",
            manager=self.manager,
        )
        self.show_animation_enabled = True  # État initial : animations activées

        # Boutons pour lancer ou revenir
        self.simulation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.85), (field_width, button_height)
            ),
            text="Lancer les Simulations",
            manager=self.manager,
        )
        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (SCREEN_WIDTH * 0.05, SCREEN_HEIGHT * 0.9),
                (SCREEN_WIDTH * 0.2, button_height),
            ),
            text="Retour",
            manager=self.manager,
        )

        # Stocker les données partagées ici
        self.shared_data = {}

    def update(self, time_delta):
        self.screen.fill(LIGHT_BACKGROUND)

        # Dessiner le titre
        title_surface = self.title_font.render(self.title_text, True, (0, 0, 0))
        title_rect = title_surface.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.1)
        )
        self.screen.blit(title_surface, title_rect)

        # Dessiner les étiquettes des champs de texte
        self._draw_label(self.num_agents_label, SCREEN_HEIGHT * 0.25)
        self._draw_label(self.alpha_label, SCREEN_HEIGHT * 0.35)
        self._draw_label(self.beta_label, SCREEN_HEIGHT * 0.45)
        self._draw_label(self.num_simulations_label, SCREEN_HEIGHT * 0.55)
        self._draw_label(self.time_limit_label, SCREEN_HEIGHT * 0.65)

        self.manager.update(time_delta)
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.simulation_button:
                    num_agents = int(self.num_agents_input.get_text())
                    alpha = float(self.alpha_input.get_text())
                    beta = float(self.beta_input.get_text())
                    num_simulations = int(self.num_simulations_input.get_text())
                    time_limit = float(self.time_limit_input.get_text())
                    show_animation = self.show_animation_enabled

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
                                    None,
                                    i + 1,
                                    show_animation,
                                    time_limit,  # Passer le temps limite
                                ),
                            )
                            processes.append(p)
                            p.start()

                        for p in processes:
                            p.join()

                        self.shared_data = dict(shared_data)
                elif event.ui_element == self.back_button:
                    return "menu"
                elif event.ui_element == self.show_animation_button:
                    # Basculer l'état des animations
                    self.show_animation_enabled = not self.show_animation_enabled
                    # Mettre à jour le texte du bouton
                    new_text = (
                        "Animations : ON"
                        if self.show_animation_enabled
                        else "Animations : OFF"
                    )
                    self.show_animation_button.set_text(new_text)
        return "simulation"

    def _draw_label(self, text, y_position):
        label_surface = self.label_font.render(text, True, (0, 0, 0))
        label_rect = label_surface.get_rect(
            midright=((SCREEN_WIDTH * 0.4) - 10, y_position + (SCREEN_HEIGHT * 0.025))
        )
        self.screen.blit(label_surface, label_rect)
