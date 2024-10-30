import pygame
from sys import exit

# Constantes
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
HOVER_COLOR_BUTTON = (200, 200, 200)

# Initialisation de Pygame (variables globales)
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mouvements de foules")
clock = pygame.time.Clock()

large_font = pygame.font.Font(None, 75)
medium_font = pygame.font.Font(None, 40)
small_font = pygame.font.Font(None, 25)


# Fonction pour créer les boutons :
def creer_bouton(text, y, x=SCREEN_WIDTH // 2, color=BLACK):
    text_surf = medium_font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(x, y))
    button_rect = text_rect.inflate(20, 20)  # Ajouter une marge autour du texte
    return text_surf, text_rect, button_rect


# Affichage du menu d'accueil
def afficher_menu():
    menu_name = large_font.render("Mouvements de foule", False, "BLACK")
    menu_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 2, 100))
    while True:
        screen.fill(WHITE)

        # Création des boutons Scenario de Simulation / Simulation personalisée / Quitter
        texte_simulation1, rect_simulation1, bouton_simulation1 = creer_bouton(
            "Scénarios de Simulation", SCREEN_HEIGHT // 2 - 50, SCREEN_WIDTH // 2
        )
        texte_simulation2, rect_simulation2, bouton_simulation2 = creer_bouton(
            "Simulation personalisée", SCREEN_HEIGHT // 2 + 10, SCREEN_WIDTH // 2
        )

        texte_quitter, rect_quitter, bouton_quitter = creer_bouton(
            "Quitter", SCREEN_HEIGHT // 2 + 80
        )

        # Obtenir la position de la souris
        mouse_pos = pygame.mouse.get_pos()

        # Dessiner le bouton "Simulation1" avec effet de survol
        if bouton_simulation1.collidepoint(mouse_pos):
            pygame.draw.rect(
                screen, HOVER_COLOR_BUTTON, bouton_simulation1
            )  # Couleur de fond si survolé
        else:
            pygame.draw.rect(
                screen, WHITE, bouton_simulation1
            )  # Fond blanc si non survolé
        pygame.draw.rect(screen, BLACK, bouton_simulation1, 2)  # Bordure noire

        # Dessiner le bouton "Simulation2" avec effet de survol
        if bouton_simulation2.collidepoint(mouse_pos):
            pygame.draw.rect(
                screen, HOVER_COLOR_BUTTON, bouton_simulation2
            )  # Couleur de fond si survolé
        else:
            pygame.draw.rect(
                screen, WHITE, bouton_simulation2
            )  # Fond blanc si non survolé
        pygame.draw.rect(screen, BLACK, bouton_simulation2, 2)  # Bordure noire

        # Dessiner le bouton "Quitter" avec effet de survol
        if bouton_quitter.collidepoint(mouse_pos):
            pygame.draw.rect(
                screen, HOVER_COLOR_BUTTON, bouton_quitter
            )  # Couleur de fond si survolé
        else:
            pygame.draw.rect(screen, WHITE, bouton_quitter)  # Fond blanc si non survolé
        pygame.draw.rect(screen, BLACK, bouton_quitter, 2)  # Bordure noire

        # Afficher le texte des boutons
        screen.blit(texte_simulation1, rect_simulation1)
        screen.blit(texte_simulation2, rect_simulation2)
        screen.blit(texte_quitter, rect_quitter)
        screen.blit(menu_name, menu_name_rect)

        pygame.display.flip()

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if bouton_simulation1.collidepoint(event.pos):
                    return "play"  # Retourner une valeur pour commencer le jeu
                elif bouton_quitter.collidepoint(event.pos):
                    pygame.quit()
                    exit()
