import pygame
import pygame_gui
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND


class MenuView:
    def __init__(self, screen):
        self.screen = screen
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Create native buttons
        self.simulation_scenario_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 200), (200, 50)),
            text="Scenarios de Simulation ",
            manager=self.manager,
        )

        self.simulation_custom_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 270), (200, 50)),
            text="Simulation customisée",
            manager=self.manager,
        )

        self.quit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((300, 340), (200, 50)),
            text="Quitter",
            manager=self.manager,
        )

    def update(self, time_delta):
        self.screen.fill(LIGHT_BACKGROUND)
        self.manager.update(time_delta)
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def handle_events(self, event):
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.simulation_scenario_button:
                    return "simulation_scenario"
                elif event.ui_element == self.simulation_custom_button:
                    return "simulation_custom"
                elif event.ui_element == self.quit_button:
                    pygame.quit()
                    exit()

        return "menu"
