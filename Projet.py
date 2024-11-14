import dataclasses
import os
from Utils import (
    pygame,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    afficher_menu,
    medium_font,
    clock,
    screen,
    small_font,
    large_font,
    afficher_scenario_simulation,
)


# Définir le chemin relatif
assets_path = os.path.join(os.path.dirname(__file__), "Assets", "test.png")

# Charger l'image
background = pygame.image.load(assets_path)

# Remplir le fond avec une couleur blanche (si nécessaire)
background.fill("WHITE")

ground = pygame.Surface((800, 400))
ground.fill("BROWN")
Player_sprite = pygame.Surface((5, 5))
Player_sprite.fill("BLACK")
player_initial_pos = 500

menu_name = medium_font.render("Mouvements de foule", False, "BLACK")


def scenario_simulation():

    return None


def scenario_perso():
    return None


# Lancer le menu d'accueil puis le jeu
if __name__ == "__main__":
    while True:
        choix = afficher_menu()
        if choix == "simulation_scenario":
            choix = afficher_scenario_simulation()
            if choix == "main":
                continue

        if choix == "simulation_perso":
            scenario_perso()
