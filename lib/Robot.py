import math
import pygame

# Two motors, alternate their speed to change direction
class Robot:
    RADIUS = 10
    def __init__(self):
        self.x, self.y = 0, 0 # coords
        self.speed1, self.speed2 = 0, 0 # motor speeds
        self.heading = 0

    def calculate_angular_speed(self):
        return (self.speed1 - self.speed2) / 2

    def calculate_speed(self):
        return (self.speed1 + self.speed2) / 2 
   
    def move(self):
        self.x += self.calculate_speed() * math.cos(self.heading)
        self.y += self.calculate_speed() * math.sin(self.heading)
        self.heading += self.calculate_angular_speed()

    def set_speed(self, speed1, speed2):
        self.speed1 = speed1
        self.speed2 = speed2
    
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.RADIUS)
        pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (self.x + 20 * math.cos(self.heading), self.y + 20 * math.sin(self.heading)), 5)
