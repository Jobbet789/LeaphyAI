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
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.RADIUS)
        pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (self.x + 20 * math.cos(self.direction), self.y + 20 * math.sin(self.direction)), 5)