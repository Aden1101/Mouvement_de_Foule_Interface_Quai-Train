import pygame
from sys import exit

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

# Fonts
large_font = pygame.font.Font(None, 75)
medium_font = pygame.font.Font(None, 40)
small_font = pygame.font.Font(None, 25)


# Classe pour gérer les zones de texte
class InputBox:
    def __init__(self, x, y, w, h, font, label_text="Paramètre", enabled=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = DARK_GRAY if enabled else DISABLED_GRAY
        self.text = ""
        self.font = font
        self.label_text = label_text
        self.enabled = enabled
        self.active = False
        self.error_message = ""

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
                if event.unicode.isdigit() or event.unicode == ".":
                    self.text += event.unicode

    def validate_text(self):
        try:
            value = float(self.text)
            if 0 <= value <= 1:
                self.error_message = ""
                print(f"Valeur acceptée pour {self.label_text}: {value}")
            else:
                self.error_message = "Veuillez entrer un nombre entre 0 et 1."
        except ValueError:
            self.error_message = "Entrée invalide. Entrez un nombre."

    def draw(self, screen):
        label_surface = self.font.render(self.label_text, True, BLACK)
        screen.blit(label_surface, (self.rect.x, self.rect.y - 25))
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        text_surface = self.font.render(
            self.text, True, WHITE if self.enabled else DARK_GRAY
        )
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + 10))

        if self.error_message:
            error_surface = small_font.render(self.error_message, True, ERROR_COLOR)
            screen.blit(error_surface, (self.rect.x, self.rect.y + 60))

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.color = DARK_GRAY if enabled else DISABLED_GRAY


# Classe pour gérer un Timer
class Timer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.start_time = 0
        self.paused_time = 0
        self.running = False
        self.paused = False

    def start(self):
        self.start_time = pygame.time.get_ticks()
        self.running = True
        self.paused = False

    def pause(self):
        if self.running and not self.paused:
            self.paused_time = pygame.time.get_ticks() - self.start_time
            self.paused = True

    def resume(self):
        if self.running and self.paused:
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
        time_text = medium_font.render(
            f"Timer: {self.get_elapsed_time():.2f}s", True, BLACK
        )
        screen.blit(time_text, (self.x, self.y))


# Fonction pour créer une seule zone de texte
def creer_input_box(x, y, width, height, label_text="Paramètre"):
    return InputBox(x, y, width, height, medium_font, label_text)


# Fonction pour créer les boutons avec un texte centré
def creer_bouton(text, y, x=SCREEN_WIDTH // 2, color=BUTTON_TEXT_COLOR):
    text_surf = medium_font.render(text, True, color)
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
    menu_name_rect = menu_name.get_rect(center=(SCREEN_WIDTH // 2, 50))

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
