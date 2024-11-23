import pygame


def creer_bouton(text, y, x=400, color=(255, 255, 255), font=None):
    if font is None:
        font = pygame.font.Font(None, 35)
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect(center=(x, y))
    button_rect = text_rect.inflate(20, 20)
    return text_surf, text_rect, button_rect


def draw_button_with_border(screen, text_surf, text_rect, button_rect, is_hovered):
    button_color = (200, 200, 200) if is_hovered else (100, 100, 100)
    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, (0, 0, 0), button_rect, 1)
    screen.blit(text_surf, text_rect)
