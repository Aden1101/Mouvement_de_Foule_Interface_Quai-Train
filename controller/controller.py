import pygame


class Controller:
    def __init__(
        self,
        menu_view,
        simulation_view,
        csv_analysis_view,
    ):
        self.menu_view = menu_view
        self.simulation_view = simulation_view
        self.csv_analysis_view = csv_analysis_view
        self.current_view = "menu"

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # Gestion des évènements

            # Pour le menu principal
            if self.current_view == "menu":
                action = self.menu_view.handle_events(event)
                if action == "simulation_scenario":
                    self.current_view = "simulation"
                elif action == "analysis_csv":
                    self.current_view = "csv_analysis"

            # Pour le menu de simulation
            elif self.current_view == "simulation":
                action = self.simulation_view.handle_events(event)
                if action == "menu":
                    self.current_view = "menu"

            # Pour le menu d'analyse statistique
            elif self.current_view == "csv_analysis":
                action = self.csv_analysis_view.handle_events(event)
                if action == "menu":
                    self.current_view = "menu"
                self.csv_analysis_view.manager.process_events(event)

            # Envoie les évènements aux "managers" de pygame pour les UI des différentes vues
            if self.current_view == "menu":
                self.menu_view.manager.process_events(event)
            elif self.current_view == "simulation":
                self.simulation_view.manager.process_events(event)
