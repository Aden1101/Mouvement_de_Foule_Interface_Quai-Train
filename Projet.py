import dataclasses
import pygame
from Utils import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    afficher_menu,
    medium_font,
    clock,
    screen,
    small_font,
    large_font,
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


# Fonction principale du jeu (avec l'animation et les éléments principaux)
def main():
    global player_initial_pos
    running = True
    while running:
        # Quitter le jeu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Affichage des éléments de jeu
        screen.blit(background, (0, 0))
        screen.blit(ground, (0, 400))
        game_name_rect = menu_name.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)
        )
        screen.blit(menu_name, game_name_rect)
        screen.blit(Player_sprite, (player_initial_pos, 300))

        # Animation du joueur
        player_initial_pos -= 5
        if player_initial_pos < 0:  # Réinitialise la position
            player_initial_pos = SCREEN_WIDTH

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

def scenario_simulation():

    while true:
        


# Lancer le menu d'accueil puis le jeu
if __name__ == "__main__":
    choix = afficher_menu()
    if choix == "play":
        main()
