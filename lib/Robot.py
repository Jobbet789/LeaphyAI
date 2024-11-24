import math
import pygame

class Robot:
    '''
    This class represents a robot in the simulation.
    The robot is a circle (for now)

    It can move, see, and be controlled by a neural network.

    The robot has two motors, one on each side, each with a speed
    This means the robot can rotate (for example by speed1 = -1 and speed2 = 1)
    
    '''
    RADIUS = 10 

    def __init__(self, x: int, y: int) -> None:
        '''
        Init the robot, get starting pos

        '''
        self.x, self.y = x, y
        self.speed1, self.speed2 = 0, 0
        self.heading = 0
        self.speedMultiplier = 10

        self.headingToBall = 0

    def calculate_speed(self) -> float:
        ''' 
        Calculate the speed of the robot based on the motor speeds
        '''
        return (self.speed1 + self.speed2) * self.speedMultiplier / 2

    def calculate_angular_speed(self) -> float:
        '''
        Calculate the angular speed of the robot based on the motor speeds 
        '''
        return (self.speed1 - self.speed2) / 2
    
    def move(self, WIDTH: int, HEIGHT: int) -> None:
        '''
        Move the robot according to the current motor speeds and heading
        '''
        speed = self.calculate_speed()
        self.x += speed * math.cos(self.heading)
        self.y += speed * math.sin(self.heading)

        self.heading += self.calculate_angular_speed()

        # CONSTRAINTS using built in function
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def vision(self, ball) -> bool:
        '''
        Get the angle to the ball if it is within the vision cone
        '''
        angle = math.atan2((ball.y + 0.5 * ball.RADIUS) - (self.y + 0.5 * self.RADIUS), (ball.x + 0.5 * ball.RADIUS) - (self.x + 0.5 * self.RADIUS))

        diff = (angle - self.heading + math.pi) % (2 * math.pi) - math.pi

        if abs(diff) < math.pi/4:
            self.headingToBall = diff
            return True
        else:
            self.headingToBall = 0
            return False
    
    def distanceToBall(self, ball: object) -> float:
        '''
        Get the distance to the ball
        '''
        return math.sqrt(((ball.x+0.5*ball.RADIUS) - (self.x+0.5*self.RADIUS))**2 + ((ball.y+0.5*ball.RADIUS) - (self.y+0.5*self.RADIUS))**2)
        
    
    def draw(self, screen) -> None:
        '''
        Draw the robot on the screen
        '''
        x, y = int(self.x), int(self.y)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), self.RADIUS)
        pygame.draw.line(screen, (0, 255, 0), (x, y), (int(self.x + 100 * math.cos(self.heading + math.pi/4)), int(self.y + 100 * math.sin(self.heading + math.pi/4))), 2)
        pygame.draw.line(screen, (0, 255, 0), (x, y), (int(self.x + 100 * math.cos(self.heading - math.pi/4)), int(self.y + 100 * math.sin(self.heading - math.pi/4))), 2)

