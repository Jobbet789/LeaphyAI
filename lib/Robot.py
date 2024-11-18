import math
import pygame

# Two motors, alternate their speed to change direction
class Robot:
    RADIUS = 10
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y # position
        self.speed1, self.speed2 = 0, 0 # motor speeds
        self.heading = 0
        self.speedMultiplier = 10
        
        self.visionLen = 150

    def calculate_angular_speed(self):
        return (self.speed1 - self.speed2) / 2

    def calculate_speed(self):
        return (self.speed1 + self.speed2) * self.speedMultiplier / 2 
   
    def move(self, WIDTH, HEIGHT):
        self.x += self.calculate_speed() * math.cos(self.heading)
        self.y += self.calculate_speed() * math.sin(self.heading)
        self.heading += self.calculate_angular_speed()

        # CONSTRAINTS
        if self.x < 0:
            self.x = 0
        if self.x > WIDTH:
            self.x = WIDTH
        if self.y < 0:
            self.y = 0
        if self.y > HEIGHT:
            self.y = HEIGHT

    def vision(self, *balls):
        angles = ()
        for ball in balls:
            # check if ball is within vision angle
            angle = math.atan2((ball.y + 0.5 * ball.RADIUS) - (self.y + 0.5 * self.RADIUS), (ball.x + 0.5 * ball.RADIUS) - (self.x + 0.5 * self.RADIUS))

            diff = (angle - self.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < math.pi/4:
                angles += (angle,)
            else: 
                angles += (False,)

        return angles
    
    def inputsNN(self, angle):
        dx = math.cos(angle)
        dy = math.sin(angle)

        vis = 1
        if angle == False:
            dx = 0
            dy = 0
            vis = 0

        dhx = math.cos(self.heading)
        dhy = math.sin(self.heading)

        return [dx, dy, dhx, dhy, vis]

    def set_speed(self, speed1, speed2):
        self.speed1 = speed1
        self.speed2 = speed2
    
    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.x), int(self.y)), self.RADIUS)
        pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (self.x + 20 * math.cos(self.heading), self.y + 20 * math.sin(self.heading)), 5)
        # draw two more lines to represent the 'vision' of the robot, 45 degrees to the left and right
        pygame.draw.line(screen, (0, 255, 255), (self.x, self.y), (self.x + self.visionLen * math.cos(self.heading + math.pi/4), self.y + self.visionLen * math.sin(self.heading + math.pi/4)), 2)
        pygame.draw.line(screen, (0, 255, 255), (self.x, self.y), (self.x + self.visionLen * math.cos(self.heading - math.pi/4), self.y + self.visionLen * math.sin(self.heading - math.pi/4)), 2)