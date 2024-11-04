import dataclasses
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

# Chargement des assets
background = pygame.image.load(
    "C:/Users/elmi_/Documents/ENPC/2A_IMI/TdLOG/Projet_Mouvement_de_foule/Mouvement_de_Foule_Interface_Quai-Train/Assets/Pontai.png"
)
background.fill("WHITE")

ground = pygame.Surface((800, 200))
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
