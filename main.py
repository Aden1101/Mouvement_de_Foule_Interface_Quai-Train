import pygame
from view.view_menu import MenuView
from view.view_simulation import SimulationView
from view.view_csv import CSVAnalysisView
from controller.controller import Controller
from model.Simulation import SimulationManager
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simulation Mouvements de Foule")

    # Création des vues
    menu_view = MenuView(screen)
    simulation_manager = SimulationManager(
        num_agents=100, barrier_width=0.15, collision_distance=0.05
    )
    simulation_view = SimulationView(screen, simulation_manager)
    csv_analysis_view = CSVAnalysisView(screen)  # Nouvelle vue d'analyse CSV

    # Création du contrôleur avec toutes les vues
    controller = Controller(
        menu_view, simulation_view, csv_analysis_view, simulation_manager
    )

    clock = pygame.time.Clock()

    # Main loop
    while True:
        time_delta = clock.tick(30) / 1000.0
        controller.handle_events()

        # Gestion des vues
        if controller.current_view == "menu":
            menu_view.update(time_delta)
        elif controller.current_view == "simulation":
            simulation_view.update(time_delta)
        elif controller.current_view == "csv_analysis":
            csv_analysis_view.update(time_delta)


if __name__ == "__main__":
    main()
