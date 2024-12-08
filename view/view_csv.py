import pygame
import pygame_gui
import matplotlib.pyplot as plt
import pandas as pd
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND


class CSVAnalysisView:
    def __init__(self, screen):
        self.screen = screen
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Titre
        self.title_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.08))
        self.title_text = "Analyse des Simulations"

        # Boutons
        button_width = SCREEN_WIDTH * 0.3
        button_height = SCREEN_HEIGHT * 0.07
        x_center = (SCREEN_WIDTH - button_width) / 2

        self.load_csv_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.4), (button_width, button_height)
            ),
            text="Charger un Fichier CSV",
            manager=self.manager,
        )
        self.show_graphs_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.5), (button_width, button_height)
            ),
            text="Afficher les Graphiques",
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

        # Stockage des données CSV
        self.dataframe = None

    def update(self, time_delta):
        self.screen.fill(LIGHT_BACKGROUND)

        # Dessiner le titre
        title_surface = self.title_font.render(self.title_text, True, (0, 0, 0))
        title_rect = title_surface.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 0.2)
        )
        self.screen.blit(title_surface, title_rect)

        self.manager.update(time_delta)
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def handle_events(self, event):
        """Gérer les événements de la vue CSV."""
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.load_csv_button:
                    self.load_csv_file()
                elif event.ui_element == self.show_graphs_button:
                    self.show_graphs()
                elif event.ui_element == self.back_button:
                    return "menu"  # Retour au menu principal
        return "csv_analysis"

    def load_csv_file(self):
        """Ouvre un dialogue pour charger un fichier CSV et le charge dans un DataFrame."""
        from tkinter import Tk, filedialog

        Tk().withdraw()  # Cache la fenêtre principale Tkinter
        file_path = filedialog.askopenfilename(
            title="Charger un Fichier CSV", filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            try:
                self.dataframe = pd.read_csv(file_path)
                print("Fichier chargé avec succès :", file_path)
                print(self.dataframe.head())
            except Exception as e:
                print("Erreur lors du chargement du fichier :", e)

    def show_graphs(self):
        """Affiche des graphiques basés sur le DataFrame chargé."""
        if self.dataframe is not None:
            try:
                import matplotlib.pyplot as plt

                # Exemple de graphique : Histogramme des temps finaux
                if "Final_time" in self.dataframe.columns:
                    plt.figure()
                    self.dataframe["Final_time"].hist(bins=10)
                    plt.title("Distribution des Temps Finaux")
                    plt.xlabel("Temps Final")
                    plt.ylabel("Nombre de Simulations")
                    plt.show()

                # Exemple de graphique : Scatter des paramètres Alpha et Beta
                if (
                    "Alpha" in self.dataframe.columns
                    and "Beta" in self.dataframe.columns
                ):
                    plt.figure()
                    plt.scatter(self.dataframe["Alpha"], self.dataframe["Beta"])
                    plt.title("Paramètres Alpha vs Beta")
                    plt.xlabel("Alpha")
                    plt.ylabel("Beta")
                    plt.show()

                # Exemple : Temps de montée et descente par simulation
                if "Simulation" in self.dataframe.columns:
                    plt.figure()
                    plt.plot(
                        self.dataframe["Simulation"],
                        self.dataframe["Blue_time"],
                        label="Temps de descente",
                        marker="o",
                    )
                    plt.plot(
                        self.dataframe["Simulation"],
                        self.dataframe["Red_time"],
                        label="Temps de montée",
                        marker="x",
                    )
                    plt.title("Temps de descente et de montée par simulation")
                    plt.xlabel("Simulation")
                    plt.ylabel("Temps")
                    plt.legend()
                    plt.show()

            except Exception as e:
                print("Erreur lors de la génération des graphiques :", e)
        else:
            print("Aucun fichier CSV chargé.")
