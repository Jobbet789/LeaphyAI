import math
import pygame

class Ball:
    RADIUS = 10
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.direction = 0
        self.speed = 0

    def move(self):
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), self.RADIUS)