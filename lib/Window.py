import math
import pygame

class Window:
    def __init__(self, WIDTH: int, HEIGHT: int):
        self.WIDTH, self.HEIGHT = WIDTH, HEIGHT
        pygame.init()

    def setup(self):
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("LeaphyAI")
        return screen
    
    def draw(self, screen, *objects):
        screen.fill((0, 0, 0))
        for obj in objects:
            obj.draw(screen)
        pygame.display.update()