import pygame
from view.view_menu import MenuView  # Vue principale
from view.view_simulation import SimulationView  # Vue menu simulation
from view.view_csv import CSVAnalysisView  # Vue menu analyse statistique
from controller.controller import Controller
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT  # Constantes de taille


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))  # Crée la fenêtre
    pygame.display.set_caption("Simulation de mouvements de foule")  # Nom de la fenêtre

    # Création des différentes vues
    menu_view = MenuView(screen)
    simulation_view = SimulationView(screen)
    csv_analysis_view = CSVAnalysisView(screen)

    # Création du contrôleur avec toutes les vues
    controller = Controller(menu_view, simulation_view, csv_analysis_view)

    clock = pygame.time.Clock()

    # Boucle principale
    while True:
        time_delta = clock.tick(30) / 1000.0
        controller.handle_events()  # Gestion des évènements

        # Gestion des vues
        if controller.current_view == "menu":
            menu_view.update(time_delta)
        elif controller.current_view == "simulation":
            simulation_view.update(time_delta)
        elif controller.current_view == "csv_analysis":
            csv_analysis_view.update(time_delta)


# Entrée programme
if __name__ == "__main__":
    main()
