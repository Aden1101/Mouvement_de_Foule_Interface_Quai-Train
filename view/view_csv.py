import pygame
import pygame_gui
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process
from model.constants import SCREEN_WIDTH, SCREEN_HEIGHT, LIGHT_BACKGROUND
from tkinter import Tk, filedialog


class CSVAnalysisView:
    def __init__(self, screen):
        self.screen = screen
        self.manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Dimensions pour les champs et boutons
        button_width = SCREEN_WIDTH * 0.3
        button_height = SCREEN_HEIGHT * 0.07
        x_center = (SCREEN_WIDTH - button_width) / 2

        # Titre
        self.title_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.08))
        self.title_text = "Analyse des Simulations"

        # Boutons
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
        self.clean_data_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (x_center, SCREEN_HEIGHT * 0.6), (button_width, button_height)
            ),
            text="Nettoyer les Données",
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
        self.file_path = None

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
                    self.launch_graphs_in_process()
                elif event.ui_element == self.clean_data_button:
                    self.launch_cleaning_in_process()
                elif event.ui_element == self.back_button:
                    return "menu"
        return "csv_analysis"

    def load_csv_file(self):
        """Ouvre un dialogue pour charger un fichier CSV et le charge dans un DataFrame."""
        Tk().withdraw()
        file_path = filedialog.askopenfilename(
            title="Charger un Fichier CSV", filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            try:
                self.file_path = file_path
                self.dataframe = pd.read_csv(file_path)
                print("Fichier chargé avec succès :", file_path)
                print(self.dataframe.head())
            except Exception as e:
                print("Erreur lors du chargement du fichier :", e)

    def launch_graphs_in_process(self):
        """Lance l'affichage des graphiques dans un processus séparé."""
        if self.dataframe is not None:
            dataframe_copy = self.dataframe.copy()
            process = Process(target=self.show_graphs_process, args=(dataframe_copy,))
            process.start()
            process.join()
        else:
            print("Aucun fichier CSV chargé pour les graphiques.")

    def launch_cleaning_in_process(self):
        """Lance le nettoyage des données dans un processus séparé."""
        if self.dataframe is not None:
            dataframe_copy = self.dataframe.copy()
            file_path = self.file_path
            process = Process(
                target=self.clean_data_process, args=(dataframe_copy, file_path)
            )
            process.start()
            process.join()
        else:
            print("Aucun fichier CSV chargé pour le nettoyage.")

    @staticmethod
    def show_graphs_process(dataframe):
        """Affiche des graphiques montrant la moyenne et la variance du temps par nombre d'agents."""
        try:
            import numpy as np  # Import explicite pour le multiprocessing

            # Vérifier les colonnes nécessaires
            if "Nb_agents" in dataframe.columns and "Final_time" in dataframe.columns:
                # Calculer la moyenne et la variance par nombre d'agents
                stats = (
                    dataframe.groupby("Nb_agents")["Final_time"]
                    .agg(["mean", "var"])
                    .reset_index()
                )
                print(stats)

                # Graphique : Moyenne du temps en fonction du nombre d'agents
                plt.figure()
                plt.plot(
                    stats["Nb_agents"],
                    stats["mean"],
                    marker="o",
                    label="Temps Moyen",
                )
                plt.title("Moyenne du Temps Final par Nombre d'Agents")
                plt.xlabel("Nombre d'Agents par Équipe")
                plt.ylabel("Temps Final Moyen")
                plt.grid(True)
                plt.legend()
                plt.show()

                # Graphique : Variance du temps en fonction du nombre d'agents
                plt.figure()
                plt.plot(
                    stats["Nb_agents"],
                    stats["var"],
                    marker="o",
                    color="red",
                    label="Variance du Temps Final",
                )
                plt.title("Variance du Temps Final par Nombre d'Agents")
                plt.xlabel("Nombre d'Agents par Équipe")
                plt.ylabel("Variance du Temps Final")
                plt.grid(True)
                plt.legend()
                plt.show()
            else:
                print("Colonnes nécessaires manquantes dans les données.")

        except Exception as e:
            print(f"Erreur lors de l'affichage des graphiques : {e}")

    @staticmethod
    def clean_data_process(dataframe, file_path):
        """Nettoie les données en supprimant les simulations incohérentes."""
        try:
            if "Final_time" not in dataframe.columns:
                print("Erreur : La colonne 'Final_time' est absente des données.")
                return

            mean_time = dataframe["Final_time"].mean()
            std_time = dataframe["Final_time"].std()
            threshold = 3  # Seuil d'écart-type

            dataframe["Outlier"] = abs(dataframe["Final_time"] - mean_time) > (
                threshold * std_time
            )
            outliers = dataframe[dataframe["Outlier"]]
            print(f"Simulations incohérentes détectées :\n{outliers}")

            cleaned_data = dataframe[~dataframe["Outlier"]].drop(columns=["Outlier"])
            cleaned_file_path = f"cleaned_{file_path.split('/')[-1]}"
            cleaned_data.to_csv(cleaned_file_path, index=False)
            print(f"Données nettoyées sauvegardées dans {cleaned_file_path}")

        except Exception as e:
            print(f"Erreur lors du nettoyage des données : {e}")
