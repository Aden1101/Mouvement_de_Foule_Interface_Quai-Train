import pygame
import pygame_gui
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND


class MenuView:
    def __init__(self, screen):
        self.screen = screen
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Titre
        self.title_font = pygame.font.Font(
            None, int(SCREEN_HEIGHT * 0.1)
        )  # Police pour le titre
        self.title_text = "Mouvements de Foules"

        # Dimensions relatives pour les boutons
        button_width = SCREEN_WIDTH * 0.3
        button_height = SCREEN_HEIGHT * 0.07
        x_center = (SCREEN_WIDTH - button_width) / 2

        # Création des boutons natifs
        self.simulation_scenario_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.4), (button_width, button_height)
            ),
            text="Scénarios de Simulation",
            manager=self.manager,
        )
        self.analysis_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.5), (button_width, button_height)
            ),
            text="Analyse de Simulations",
            manager=self.manager,
        )
        self.quit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.6), (button_width, button_height)
            ),
            text="Quitter",
            manager=self.manager,
        )

    def update(self, time_delta):
        self.screen.fill(LIGHT_BACKGROUND)

        # Afficher le titre
        title_surface = self.title_font.render(self.title_text, True, (0, 0, 0))
        title_rect = title_surface.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.2)
        )
        self.screen.blit(title_surface, title_rect)

        self.manager.update(time_delta)  # Gestion des interactions
        self.manager.draw_ui(self.screen)  # Affiche le reste de l'UI (boutons,etc...)
        pygame.display.flip()  # Mets à jour l'affichage

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.simulation_scenario_button:
                    return "simulation_scenario"
                elif event.ui_element == self.analysis_button:
                    return "analysis_csv"
                elif event.ui_element == self.quit_button:
                    pygame.quit()
                    exit()
        return "menu"
