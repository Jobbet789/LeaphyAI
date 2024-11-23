import math
import pygame

class Ball:
    RADIUS = 10
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.RADIUS)