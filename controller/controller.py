import pygame


class Controller:
    def __init__(self, menu_view, simulation_view, simulation_manager):
        self.menu_view = menu_view
        self.simulation_view = simulation_view
        self.simulation_manager = simulation_manager
        self.current_view = "menu"

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # Handle events based on the current view
            if self.current_view == "menu":
                action = self.menu_view.handle_events(event)
                if action == "simulation_scenario":
                    self.current_view = "simulation"
                elif action == "simulation_custom":
                    pass  # Add logic for custom simulation
            elif self.current_view == "simulation":
                action = self.simulation_view.handle_events(event)
                if action == "menu":
                    self.current_view = "menu"

            # Pass events to pygame_gui
            if self.current_view == "menu":
                self.menu_view.manager.process_events(event)
            elif self.current_view == "simulation":
                self.simulation_view.manager.process_events(event)
