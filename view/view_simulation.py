import pygame
import pygame_gui
from multiprocessing import Process, Manager
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND
from model.animation import run_simulation


class SimulationView:
    def __init__(self, screen, simulation_manager):
        self.screen = screen
        self.simulation_manager = simulation_manager
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Titre
        self.title_font = pygame.font.Font(
            None, int(SCREEN_HEIGHT * 0.08)
        )  # Police pour le titre
        self.title_text = "Simulation Personnalisée"

        # Police pour les étiquettes
        self.label_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.04))

        # Dimensions et position centrée pour les champs et les boutons
        field_width = SCREEN_WIDTH * 0.4  # Largeur des champs et boutons
        field_height = SCREEN_HEIGHT * 0.05  # Hauteur des champs
        label_width = SCREEN_WIDTH * 0.15  # Largeur des étiquettes
        button_height = SCREEN_HEIGHT * 0.07  # Hauteur des boutons
        x_center = (SCREEN_WIDTH - field_width) / 2  # Centrage horizontal
        label_x = (
            x_center - label_width - 20
        )  # Position des étiquettes à gauche des champs

        # Champs de saisie pour les paramètres
        self.num_agents_label = "Nombre d'agents :"
        self.num_agents_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.25), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.num_agents_input.set_text("100")

        self.barrier_width_label = "Largeur de la barrière :"
        self.barrier_width_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.35), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.barrier_width_input.set_text("0.15")

        self.g_label = "Attraction (g) :"
        self.g_input = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.45), (field_width, field_height)
            ),
            manager=self.manager,
        )
        self.g_input.set_text("0.1")

        # Boutons pour l'interface
        self.simulation_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.6), (field_width, button_height)
            ),
            text="Lancer la Simulation",
            manager=self.manager,
        )
        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (SCREEN_WIDTH * 0.05, SCREEN_HEIGHT * 0.85),
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
        self._draw_label(self.barrier_width_label, SCREEN_HEIGHT * 0.35)
        self._draw_label(self.g_label, SCREEN_HEIGHT * 0.45)

        self.manager.update(time_delta)
        self.manager.draw_ui(self.screen)

        # Afficher les données après simulation
        if self.shared_data:
            font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.03))
            text = font.render(f"Résultat : {self.shared_data}", True, (0, 0, 0))
            self.screen.blit(text, (SCREEN_WIDTH * 0.05, SCREEN_HEIGHT * 0.7))

        pygame.display.flip()

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.simulation_button:
                    # Lire les paramètres
                    num_agents = int(self.num_agents_input.get_text())
                    barrier_width = float(self.barrier_width_input.get_text())
                    g = float(self.g_input.get_text())

                    # Lancer la simulation
                    with Manager() as manager:
                        shared_data = manager.dict()
                        p = Process(
                            target=run_simulation,
                            args=(shared_data, num_agents, barrier_width, g),
                        )
                        p.start()
                        p.join()

                        # Récupérer les données après simulation
                        self.shared_data = dict(shared_data)
                elif event.ui_element == self.back_button:
                    return "menu"
        return "simulation"

    def _draw_label(self, text, y_position):
        """Dessiner une étiquette à une position donnée."""
        label_surface = self.label_font.render(text, True, (0, 0, 0))
        label_rect = label_surface.get_rect(
            midright=((SCREEN_WIDTH * 0.4) - 10, y_position + (SCREEN_HEIGHT * 0.025))
        )
        self.screen.blit(label_surface, label_rect)
