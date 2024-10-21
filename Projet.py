import dataclasses
import pygame
import Individual
import Simulation
from sys import exit


@dataclasses.dataclass
class Player:
    def __init__(
        self,
        name,
    ):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name


pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Mouvements de foules")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 50)


background = pygame.image.load(
    "C:/Users/elmi_/Documents/ENPC/2A_IMI/TdLOG/Projet/Assets/Pontai.png"
)

background.fill("WHITE")

ground = pygame.Surface((800, 200))
ground.fill("BROWN")
Player_sprite = pygame.Surface((5, 5))
Player_sprite.fill("BLACK")
player_initial_pos = 500

game_name = font.render("Mouvements de foule", False, "BLACK")

while True:
    # Quit the game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    screen.blit(background, (0, 0))
    screen.blit(ground, (0, 400))
    screen.blit(game_name, (230, 200))
    screen.blit(Player_sprite, (player_initial_pos, 300))
    player_initial_pos -= 5

    pygame.display.update()
    clock.tick(30)
