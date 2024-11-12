import pygame
from sys import exit
import os

# Constantes
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BACKGROUND = (230, 230, 230)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (100, 100, 100)
BUTTON_TEXT_COLOR = WHITE
ERROR_COLOR = (255, 50, 50)
DISABLED_GRAY = (200, 200, 200)

# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mouvements de foules")
clock = pygame.time.Clock()

# Chemin relatif de l'image d'icône
icon_path = os.path.join(
    "Mouvement_de_Foule_Interface_Quai-Train", "Assets", "Pontai.png"
)

# Chargement de l'image de l'icône
icon_image = pygame.image.load(icon_path)

# Définir l'image comme icône de la fenêtre
pygame.display.set_icon(icon_image)

# Fonts
large_font = pygame.font.Font(None, 75)
medium_font = pygame.font.Font(None, 35)
small_font = pygame.font.Font(None, 25)
vsmall_font = pygame.font.Font(None, 15)


# Classe pour gérer les zones de texte
class InputBox:
    def __init__(
        self,
        x,
        y,
        w=55,
        h=25,
        font=small_font,
        label_text="Paramètre",
        enabled=True,
        min_value=0.0,
        max_value=1.0,
    ):
        self.width = w
        self.height = h
        self.rect = pygame.Rect(x, y, w, h)
        self.color = DARK_GRAY if enabled else DISABLED_GRAY
        self.text = ""
        self.font = small_font
        self.label_text = label_text
        self.enabled = enabled
        self.active = False
        self.error_message = ""
        self.min_value = min_value  # Valeur minimale acceptable
        self.max_value = max_value  # Valeur maximale acceptable
        self.max_length = 4  # Longueur maximale de saisie

    def handle_event(self, event):
        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = LIGHT_GRAY if self.active else DARK_GRAY

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.validate_text()
            else:
                # Ajout de caractères uniquement si la longueur maximale n'est pas atteinte
                if len(self.text) < self.max_length and (
                    event.unicode.isdigit() or event.unicode == "."
                ):
                    self.text += event.unicode

    def validate_text(self):
        """Vérifie que le texte entré est un nombre flottant et qu'il est dans l'intervalle [min_value, max_value]."""
        try:
            value = float(self.text)
            if self.min_value <= value <= self.max_value:
                self.error_message = ""
                print(f"Valeur acceptée pour {self.label_text}: {value}")
            else:
                self.error_message = (
                    f"Entrez un nombre entre {self.min_value} et {self.max_value}."
                )
        except ValueError:
            self.error_message = "Entrée invalide. Entrez un nombre."

    def draw(self, screen):
        label_surface = self.font.render(self.label_text, True, BLACK)
        screen.blit(label_surface, (self.rect.x, self.rect.y + self.height * 1.5))
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = self.font.render(
            self.text, True, WHITE if self.enabled else DARK_GRAY
        )
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + self.height / 4))

        if self.error_message:
            error_surface = small_font.render(self.error_message, True, ERROR_COLOR)
            screen.blit(error_surface, (self.rect.x, self.rect.y + 60))

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.color = DARK_GRAY if enabled else DISABLED_GRAY


# Classe pour gérer un Timer
class Timer:
    def __init__(self, x, y, color, font=small_font):
        self.x = x
        self.y = y
        self.start_time = 0
        self.paused_time = 0
        self.running = False
        self.paused = False
        self.color = color
        self.font = font

    def start(self):
        self.start_time = pygame.time.get_ticks()
        self.running = True
        self.paused = False

    def pause(self):
        if self.running and not self.paused:
            self.paused_time = pygame.time.get_ticks() - self.start_time
            self.paused = True
        elif self.running and self.paused:
            self.start_time = pygame.time.get_ticks() - self.paused_time
            self.paused = False

    def get_elapsed_time(self):
        if self.running:
            if self.paused:
                return self.paused_time / 1000
            else:
                return (pygame.time.get_ticks() - self.start_time) / 1000
        return 0

    def draw(self, screen):
        time_text = self.font.render(
            f"Timer: {self.get_elapsed_time():.2f}s", True, self.color
        )
        screen.blit(time_text, (self.x, self.y))


# Fonction pour créer les boutons
def creer_bouton(
    text, y, x=SCREEN_WIDTH // 2, color=BUTTON_TEXT_COLOR, font=medium_font
):
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(x, y))
    button_rect = text_rect.inflate(20, 20)
    return text_surf, text_rect, button_rect


# Fonction pour dessiner un bouton avec une bordure fine
def draw_button_with_border(text_surf, text_rect, button_rect, is_hovered):
    button_color = LIGHT_GRAY if is_hovered else DARK_GRAY
    pygame.draw.rect(screen, button_color, button_rect, border_radius=8)
    pygame.draw.rect(screen, BLACK, button_rect, 1, border_radius=8)
    screen.blit(text_surf, text_rect)


# Affichage du menu d'accueil avec une zone de texte de test
def afficher_menu():
    # Titre du menu
    menu_name = large_font.render("Mouvements de foule", True, BLACK)
    menu_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 2, 100))

    # Création des boutons du menu
    texte_simulation1, rect_simulation1, bouton_simulation1 = creer_bouton(
        "Scénarios de Simulation", SCREEN_HEIGHT // 2 - 50, SCREEN_WIDTH // 2
    )
    texte_simulation2, rect_simulation2, bouton_simulation2 = creer_bouton(
        "Simulation personnalisée", SCREEN_HEIGHT // 2 + 10, SCREEN_WIDTH // 2
    )
    texte_quitter, rect_quitter, bouton_quitter = creer_bouton(
        "Quitter", SCREEN_HEIGHT // 2 + 80
    )

    while True:
        screen.fill(LIGHT_BACKGROUND)

        # Afficher le titre
        screen.blit(menu_name, menu_name_rect)

        # Gérer les événements pour la zone de texte, boutons et le timer
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            # Boutons pour le menu et le timer
            if event.type == pygame.MOUSEBUTTONDOWN:
                if bouton_simulation1.collidepoint(event.pos):
                    return "simulation_scenario"
                    print("Scénarios de Simulation sélectionné")
                elif bouton_simulation2.collidepoint(event.pos):
                    return "simulation_perso"
                    print("Simulation personnalisée sélectionnée")
                elif bouton_quitter.collidepoint(event.pos):
                    pygame.quit()
                    exit()
        # Obtenir la position de la souris pour l'effet de survol
        mouse_pos = pygame.mouse.get_pos()

        # Dessiner les boutons avec effet de survol et bordure fine
        draw_button_with_border(
            texte_simulation1,
            rect_simulation1,
            bouton_simulation1,
            bouton_simulation1.collidepoint(mouse_pos),
        )
        draw_button_with_border(
            texte_simulation2,
            rect_simulation2,
            bouton_simulation2,
            bouton_simulation2.collidepoint(mouse_pos),
        )
        draw_button_with_border(
            texte_quitter,
            rect_quitter,
            bouton_quitter,
            bouton_quitter.collidepoint(mouse_pos),
        )

        pygame.display.flip()
        clock.tick(30)


# Affichage du menu d'accueil avec une zone de texte de test
def afficher_scenario_simulation():
    # Titre :
    menu_name = medium_font.render("Scenarios de Simulation", True, BLACK)
    menu_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 2, 20))

    quai_name = medium_font.render("QUAI", True, BLACK)
    quai_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 3, 140))
    train_name = medium_font.render("TRAIN", True, BLACK)
    train_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 1.7, 140))

    timer_simulation = Timer(10, 50, WHITE, medium_font)
    timer_simulation.start()

    # Création des boutons du menu
    texte_timer, rect_timer, bouton_timer = creer_bouton(
        "Pause", 62, timer_simulation.x + 200, font=small_font
    )

    param1 = InputBox(
        255, 50, max_value=1, min_value=0, font=vsmall_font, label_text="P1"
    )
    param2 = InputBox(
        param1.width + 260,
        50,
        max_value=10,
        min_value=0,
        font=vsmall_font,
        label_text="P2",
    )

    texte_accueil, rect_accueil, bouton_accueil = creer_bouton(
        "Menu Principal", SCREEN_HEIGHT - 50, SCREEN_WIDTH - 200
    )

    while True:
        screen.fill(LIGHT_BACKGROUND)

        # Afficher les textes
        screen.blit(menu_name, menu_name_rect)
        screen.blit(quai_name, quai_name_rect)
        screen.blit(train_name, train_name_rect)

        # Gérer les événements pour la zone de texte, boutons et le timer
        for event in pygame.event.get():

            param1.handle_event(event)
            param2.handle_event(event)

            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            # Boutons pour le menu et le timer
            if event.type == pygame.MOUSEBUTTONDOWN:
                if bouton_accueil.collidepoint(event.pos):
                    return "main"
                if bouton_timer.collidepoint(event.pos):
                    timer_simulation.pause()

        # Obtenir la position de la souris pour l'effet de survol
        mouse_pos = pygame.mouse.get_pos()

        pygame.draw.rect(screen, LIGHT_GRAY, (0, 40, SCREEN_WIDTH, 45))
        pygame.draw.line(screen, BLACK, (400, 40), (400, 84), 2)
        pygame.draw.line(screen, BLACK, (250, 40), (250, 84), 2)
        pygame.draw.line(
            screen,
            BLACK,
            (SCREEN_WIDTH / 3, SCREEN_HEIGHT / (4.5)),
            (SCREEN_WIDTH / 3, SCREEN_HEIGHT * 2 / (4.5)),
            4,
        )

        pygame.draw.line(
            screen,
            BLACK,
            (SCREEN_WIDTH / 3, SCREEN_HEIGHT * 2 / (4.5) + 100),
            (SCREEN_WIDTH / 3, SCREEN_HEIGHT * 4 / (4.5)),
            4,
        )

        # Dessiner les boutons avec effet de survol et bordure fine

        draw_button_with_border(
            texte_accueil,
            rect_accueil,
            bouton_accueil,
            bouton_accueil.collidepoint(mouse_pos),
        )

        draw_button_with_border(
            texte_timer,
            rect_timer,
            bouton_timer,
            bouton_timer.collidepoint(mouse_pos),
        )

        param1.draw(screen)
        param2.draw(screen)
        timer_simulation.draw(screen)
        pygame.display.flip()
        clock.tick(30)
