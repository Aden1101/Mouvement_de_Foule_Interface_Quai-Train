import pygame
import pygame_gui
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND


class SimulationView:
    def __init__(self, screen, simulation_manager):
        self.screen = screen
        self.simulation_manager = simulation_manager
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Add "Back" button
        self.back_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((20, SCREEN_HEIGHT - 60), (150, 40)),
            text="Menu Principal",
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
                if event.ui_element == self.back_button:
                    return "menu"  # Return to main menu

        return "simulation"  # Stay on the simulation screen
