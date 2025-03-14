import pygame
import sys
import math
import random

class PhysicsObject:
    """Base class for objects with physics"""
    def __init__(self, x, y, radius, color, friction=0.01):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vel_x = 0
        self.vel_y = 0
        self.mass = radius * radius  # Mass proportional to size
        self.friction = friction

    def update(self, dt, walls):
        # Apply friction
        self.vel_x *= (1 - self.friction)
        self.vel_y *= (1 - self.friction)
        
        # Basic physics update
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self._handle_wall_collision(walls)

    def _handle_wall_collision(self, walls):
        """Handle collision with walls"""
        # Bounce off walls
        if self.x - self.radius < walls.thickness:
            self.x = walls.thickness + self.radius
            self.vel_x = -self.vel_x * 0.8  # Damping factor
        elif self.x + self.radius > walls.width - walls.thickness:
            self.x = walls.width - walls.thickness - self.radius
            self.vel_x = -self.vel_x * 0.8
            
        if self.y - self.radius < walls.thickness:
            self.y = walls.thickness + self.radius
            self.vel_y = -self.vel_y * 0.8
        elif self.y + self.radius > walls.height - walls.thickness:
            self.y = walls.height - walls.thickness - self.radius
            self.vel_y = -self.vel_y * 0.8

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, other):
        # Distance between centers
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if colliding
        return distance < self.radius + other.radius

    def resolve_collision(self, other):
        # Calculate collision normal
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:  # Prevent division by zero
            nx, ny = 1, 0
        else:
            nx, ny = dx / distance, dy / distance
            
        # Relative velocity
        rel_vel_x = other.vel_x - self.vel_x
        rel_vel_y = other.vel_y - self.vel_y
        
        # Relative velocity in the normal direction
        normal_vel = rel_vel_x * nx + rel_vel_y * ny
        
        # Don't resolve if objects are moving away from each other
        if normal_vel > 0:
            return
            
        # Collision impulse
        restitution = 0.8  # Bounciness factor
        impulse = (-(1 + restitution) * normal_vel) / (1/self.mass + 1/other.mass)
        
        # Apply impulse
        self.vel_x -= (impulse * nx) / self.mass
        self.vel_y -= (impulse * ny) / self.mass
        other.vel_x += (impulse * nx) / other.mass
        other.vel_y += (impulse * ny) / other.mass
        
        # Adjust positions to prevent sticking
        overlap = (self.radius + other.radius - distance) * 0.5
        self.x -= overlap * nx
        self.y -= overlap * ny
        other.x += overlap * nx
        other.y += overlap * ny

    def is_in_target_area(self, target):
        """Check if object is in the target area"""
        return (self.x > target.x and self.y > target.y)


class Robot(PhysicsObject):
    def __init__(self, x, y):
        super().__init__(x, y, 25, (100, 100, 100), friction=0.05)
        self.rotation = 0  # In radians
        self.motor_speeds = [0, 0]  # Left and right motor speeds
        self.angular_velocity = 0
        self.speed_multiplier = 0.5  # Reduce speed

    def set_action(self, action):
        # Update motor speeds based on action
        # Clamp values between -1 and 1
        self.motor_speeds[0] = max(-1, min(1, action[0])) * 100  # Scale to original range
        self.motor_speeds[1] = max(-1, min(1, action[1])) * 100  # Scale to original range
    
    def calculate_speed(self):
        # Average of the two motor speeds determines forward speed
        return (self.motor_speeds[0] + self.motor_speeds[1]) * self.speed_multiplier
    
    def calculate_angular_speed(self):
        # Difference between motor speeds determines rotation
        return (self.motor_speeds[1] - self.motor_speeds[0]) * 0.08
    
    def update(self, dt, walls):
        # Calculate speeds
        speed = self.calculate_speed()
        self.angular_velocity = self.calculate_angular_speed()
        
        # Update rotation
        self.rotation += self.angular_velocity * dt
        
        # Compute forward direction from current rotation
        direction_x = math.cos(self.rotation)
        direction_y = math.sin(self.rotation)
        
        # Update velocities
        self.vel_x = direction_x * speed
        self.vel_y = direction_y * speed
        
        # Update position using parent method (which includes friction)
        super().update(dt, walls)
    
    def draw(self, screen):
        # Draw the robot body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw a line to show the orientation
        line_end_x = self.x + math.cos(self.rotation) * self.radius
        line_end_y = self.y + math.sin(self.rotation) * self.radius
        pygame.draw.line(screen, (0, 0, 0), (int(self.x), int(self.y)), 
                         (int(line_end_x), int(line_end_y)), 3)


class Ball(PhysicsObject):
    def __init__(self, x, y, color, radius=15):
        super().__init__(x, y, radius, color, friction=0.03)


class BallFactory:
    @staticmethod
    def create_pool_formation(center_x, center_y, radius=15, ball_count=3):
        """Create 3 balls in a triangular pool-like formation"""
        # Define the distance between ball centers
        spacing = radius * 2.2  # Slightly more than 2 radii for a small gap

        balls = [Ball(center_x, center_y - spacing/2, (255, 0, 0)),
                Ball(center_x - spacing/2, center_y + spacing/2, (0, 255, 0)),
                Ball(center_x + spacing/2, center_y + spacing/2, (0, 0, 255))]
        
        random.shuffle(balls) 

        # Keep ball_count balls
        balls = balls[:ball_count]

        return balls


class Walls:
    def __init__(self, width, height, thickness):
        self.width = width
        self.height = height
        self.thickness = thickness
        self.color = (0, 0, 0)
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (0, 0, self.width, self.thickness))
        pygame.draw.rect(screen, self.color, (0, 0, self.thickness, self.height))
        pygame.draw.rect(screen, self.color, (0, self.height - self.thickness, self.width, self.thickness))
        pygame.draw.rect(screen, self.color, (self.width - self.thickness, 0, self.thickness, self.height))


class Target:
    def __init__(self, width, height, thickness, size):
        self.size = size
        self.x = width - thickness - size
        self.y = height - thickness - size
        self.color = (255, 240, 200)
        self.center_x = self.x + size/2
        self.center_y = self.y + size/2
    
    def draw(self, screen):
        # Draw target area
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))
        
        # Draw a border for the target area
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.size, self.size), 2)


class RewardSystem:
    def __init__(self, target, max_distance):
        self.target = target
        self.max_distance = max_distance
        self.prev_ball_positions = []
        self.prev_robot_position = None
        self.removed_balls = [] # Add 0, 0 when doing one ball1
    
    def initialize(self, balls, robot):
        self.prev_ball_positions = [(ball.x, ball.y) for ball in balls]
        self.prev_robot_position = (robot.x, robot.y)
        self.removed_balls = []
    
    def calculate(self, balls, robot, all_objects):
        """Calculate reward based on the current state"""
        reward = 0
    
        # Small time penalty to encourage efficiency
        reward -= 0.01
        
        # Target center coordinates
        target_center_x = self.target.center_x
        target_center_y = self.target.center_y

        # Current robot center coordinates
        robot_center_x = robot.x + robot.radius
        robot_center_y = robot.y + robot.radius

        # Ball rewards calculation
        balls_to_remove = []

        moving_towards_any_ball = False
        
        for i, ball in enumerate(balls):
            # Skip balls that are already removed
            if ball in self.removed_balls:
                continue
                
            # Check if ball is in target area
            in_target = ball.is_in_target_area(self.target)
            
            if in_target:
                # Give a one-time reward for getting the ball in the target area
                reward += 200
                balls_to_remove.append(ball)
            else:
                # Distance-based reward component for balls not yet in target
                current_dist = math.sqrt((ball.x - target_center_x)**2 + (ball.y - target_center_y)**2)
                prev_dist = math.sqrt((self.prev_ball_positions[i][0] - target_center_x)**2 + 
                                    (self.prev_ball_positions[i][1] - target_center_y)**2)
                
                # Reward for moving toward target
                if prev_dist - current_dist > 0:  # If the ball is moving towards the target
        # Keep only i-th ball
                    reward += 0.03

                current_dist_robot = math.sqrt((ball.x - (robot.x + robot.radius))**2 + (ball.y - (robot.y + robot.radius))**2)
                prev_dist_robot = math.sqrt((self.prev_ball_positions[i][0] - (self.prev_robot_position[0] + robot.radius))**2 + 
                                    (self.prev_ball_positions[i][1] - (self.prev_robot_position[1] + robot.radius))**2)

                # Reward for moving towards any ball
                if prev_dist_robot - current_dist_robot > 0:
                    moving_towards_any_ball = True

        if moving_towards_any_ball:
            reward += 0.02

        # Store current robot position for next calculation
        self.prev_robot_position = (robot.x, robot.y)
        
        # Update previous positions for next calculation
        self.prev_ball_positions = [(ball.x, ball.y) for ball in balls]

        
        # Remove balls that entered the target area
        for ball in balls_to_remove:
            if ball in balls and ball not in self.removed_balls:
                self.removed_balls.append(ball)
                if ball in all_objects:
                    all_objects.remove(ball)

        if self.check_completion(balls):
            reward += 300
        
        return reward
    
    def check_completion(self, balls):
        """Check if the task is complete (all balls in target)"""
        return len(balls) == len(self.removed_balls) or all(ball in self.removed_balls for ball in balls)

    
    def get_removed_balls_count(self):
        return len(self.removed_balls)


class PhysicsEngine:
    @staticmethod
    def update_objects(objects, dt, walls):
        # Update all objects
        for obj in objects:
            obj.update(dt, walls)
        
        # Check for collisions between all objects
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if objects[i].check_collision(objects[j]):
                    objects[i].resolve_collision(objects[j])


class StateGenerator:
    @staticmethod
    def get_state(robot, balls, target, walls, removed_balls):
        """Return state representation for agent"""
        state = []
        
        # Calculate angle to corner
        dx_corner = target.center_x - robot.x
        dy_corner = target.center_y - robot.y
        angle_to_corner = math.atan2(dy_corner, dx_corner)
        # Scale to [-1, 1]
        angle_to_corner_scaled = angle_to_corner / math.pi
        state.append(angle_to_corner_scaled)
            
            
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # sort balls by color
        balls = sorted(balls, key=lambda x: colors.index(x.color)) 
        
        # Calculate angles to each ball
        for ball in balls:
            if ball in removed_balls:
                # If ball is removed, use a default value to indicate it's in the target
                state.append(0)  # Neutral angle value when ball is removed
            else:
                dx_ball = ball.x - robot.x
                dy_ball = ball.y - robot.y
                angle_to_ball = math.atan2(dy_ball, dx_ball)
                # Scale to [-1, 1]
                angle_to_ball_scaled = angle_to_ball / math.pi
                state.append(angle_to_ball_scaled)
        
        """
        Order of colors of balls is (255, 0 , 0), (0, 255, 0), (0, 0, 255)
        If the length of balls is 1, add 0, 0 for the other balls in the correct order
            If ball.color == (255, 0, 0), add 0, 0
            If ball.color == (0, 255, 0), add 0 before the last item in the list, add 0 after
            If ball.color == (0, 0, 255), add 0, 0 before the last item in the list

        If the length of balls is 2, add 0 for the other ball in the correct order
        """

        if len(balls) == 1:
            if balls[0].color == (255, 0, 0):
                state.append(0)
                state.append(0)
            elif balls[0].color == (0, 255, 0):
                state.insert(-1, 0)
                state.append(0)
            elif balls[0].color == (0, 0, 255):
                state.insert(-1, 0)
                state.insert(-1, 0)
        
        if len(balls) == 2:
            # Determine which color is missing in balls
            missing_color = None
            for color in colors:
                if color not in [ball.color for ball in balls]:
                    missing_color = color
                    break
            
            if missing_color == (255, 0, 0):
                state.insert(-2, 0)
            elif missing_color == (0, 255, 0):
                state.insert(-1, 0)
            elif missing_color == (0, 0, 255):
                state.append(0)
        

        

        # Add robot position scaled to [-1, 1]
        # Scale x from [WALL_THICKNESS, WIDTH-WALL_THICKNESS] to [-1, 1]
        x_scaled = 2 * (robot.x - walls.thickness) / (walls.width - 2 * walls.thickness) - 1
        # Scale y from [WALL_THICKNESS, HEIGHT-WALL_THICKNESS] to [-1, 1]
        y_scaled = 2 * (robot.y - walls.thickness) / (walls.height - 2 * walls.thickness) - 1
        state.append(x_scaled)
        state.append(y_scaled)
        
        # Add robot orientation scaled to [-1, 1]
        orientation_scaled = robot.rotation / math.pi
        state.append(orientation_scaled)
        
        return state


class UI:
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont(None, 24)
    
    def draw(self, target, walls, all_objects, total_reward, balls, reward_system, done):
        self.screen.fill((255, 255, 255))
        
        # Draw target and walls
        target.draw(self.screen)
        walls.draw(self.screen)
        
        # Draw all objects
        for obj in all_objects:
            obj.draw(self.screen)
        
        # Draw instructions
        controls_text = "Controls: W,A,S,D to move | R to reset"
        text_surface = self.font.render(controls_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (20, self.height - 40))
        
        # Draw current reward
        reward_text = f"Total Reward: {total_reward:.2f}"
        reward_surface = self.font.render(reward_text, True, (0, 0, 0))
        self.screen.blit(reward_surface, (20, self.height - 70))
        
        # Draw status of balls in target
        status_text = f"Balls in target: {reward_system.get_removed_balls_count()}/{len(balls)}"
        status_surface = self.font.render(status_text, True, (0, 0, 0))
        self.screen.blit(status_surface, (20, self.height - 100))
        
        # Draw completion status
        if done:
            completion_text = "TASK COMPLETE!"
            completion_surface = self.font.render(completion_text, True, (0, 255, 0))
            text_rect = completion_surface.get_rect(center=(self.width//2, 50))
            self.screen.blit(completion_surface, text_rect)
        
        pygame.display.flip()


class Game:
    def __init__(self, rendered=True, physics_steps=1):
        # Constants
        self.WIDTH, self.HEIGHT = 800, 600
        self.FPS = 60
        self.WALL_THICKNESS = 20
        self.TARGET_SIZE = 150
        self.MAX_DISTANCE = math.sqrt(self.WIDTH**2 + self.HEIGHT**2)
        
        # Game state
        self.rendered = rendered
        self.physics_steps = physics_steps
        self.running = True
        self.clock = pygame.time.Clock()
        self.dt = 1.0 / self.FPS
        self.total_reward = 0
        self.done = False
        
        # Initialize pygame
        pygame.init()
        
        # Initialize components
        self.walls = Walls(self.WIDTH, self.HEIGHT, self.WALL_THICKNESS)
        self.target = Target(self.WIDTH, self.HEIGHT, self.WALL_THICKNESS, self.TARGET_SIZE)
        
        # Initialize pygame window only if in rendered mode
        if self.rendered:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Robot and Balls Physics Simulation")
            self.ui = UI(self.screen, self.WIDTH, self.HEIGHT)
        
        # Initialize simulation
        self.reset_simulation()
    
    def reset_simulation(self, ball_count=3):
        # Generate random position for robot in the left third of the screen
        robot_x = random.randint(self.WALL_THICKNESS, self.WIDTH // 3)
        robot_y = random.randint(self.WALL_THICKNESS, self.HEIGHT - self.WALL_THICKNESS)
        """
        # TEMP ##
        if random.random() < 0.5:
            robot_x = random.randint(self.WALL_THICKNESS, self.WIDTH // 4)
        else:
            robot_x = random.randint(self.WIDTH // 4 * 3, self.WIDTH - self.WALL_THICKNESS)

        if random.random() < 0.5:
            robot_y = random.randint(self.WALL_THICKNESS, self.HEIGHT // 4)
        else:
            robot_y = random.randint(self.HEIGHT // 4 * 3, self.HEIGHT - self.WALL_THICKNESS)
        """

        # Reset robot
        self.robot = Robot(robot_x, robot_y)
        self.robot.rotation = random.uniform(0, 2 * math.pi)
        
        # Create balls in formation
        self.balls = BallFactory.create_pool_formation(self.WIDTH * 3 // 4, self.HEIGHT // 2, ball_count=ball_count)

        # Keep a random 1 ball from self.balls, overrite self.balls
        # ball = random.choice(self.balls)
        #self.balls = [ball]
        
        # Update object list
        self.all_objects = [self.robot] + self.balls
        
        # Initialize reward system
        self.reward_system = RewardSystem(self.target, self.MAX_DISTANCE)
        self.reward_system.initialize(self.balls, self.robot)
        
        # Reset tracking variables
        self.total_reward = 0
        self.done = False
    
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Add reset functionality
                    self.reset_simulation()
        
        # Process keyboard input for robot control
        keys = pygame.key.get_pressed()
        left_motor = 0
        right_motor = 0
        
        if keys[pygame.K_w]:  # Forward
            left_motor = 1.0
            right_motor = 1.0
        if keys[pygame.K_s]:  # Backward
            left_motor = -1.0
            right_motor = -1.0
        if keys[pygame.K_a]:  # Turn left
            left_motor -= 0.5
            right_motor += 0.5
        if keys[pygame.K_d]:  # Turn right
            left_motor += 0.5
            right_motor -= 0.5
            
        # Apply the action directly
        _, reward, done = self.action(left_motor, right_motor)
        
        # Update total reward
        if self.rendered:
            self.total_reward += reward
    
    def action(self, motor_left, motor_right):
        """Apply motor commands and step the simulation"""
        # Set robot action
        self.robot.set_action([motor_left, motor_right])

        total_reward = 0
        
        # Run multiple physics steps per action
        for _ in range(self.physics_steps):
            # In rendered mode, get the actual time delta
            if self.rendered:
                self.dt = self.clock.tick(self.FPS) / 1000.0
            
            # Update physics
            PhysicsEngine.update_objects(self.all_objects, self.dt, self.walls)
            
            # Calculate reward
            step_reward = self.reward_system.calculate(self.balls, self.robot, self.all_objects)
            total_reward += step_reward
            
            # Check if episode is done
            if not self.done:
                self.done = self.reward_system.check_completion(self.balls)
                if self.done:
                    break
        
        # Return state, reward, and done flag
        return self.get_state(), total_reward, self.done
    
    def get_state(self):
        """Return state representation for agent"""
        return StateGenerator.get_state(
            self.robot, 
            self.balls, 
            self.target, 
            self.walls, 
            self.reward_system.removed_balls
        )
    
    def update(self):
        """Process events and handle keyboard input (for rendered mode)"""
        if self.rendered:
            self.process_events()
    
    def draw(self):
        """Render the scene (for rendered mode)"""
        if not self.rendered:
            return
        
        self.ui.draw(
            self.target, 
            self.walls, 
            self.all_objects, 
            self.total_reward, 
            self.balls, 
            self.reward_system, 
            self.done
        )
    
    def run(self):
        """Run the game loop (only used in rendered mode)"""
        if not self.rendered:
            return
            
        while self.running:
            self.update()  # Process input and update physics
            self.draw()    # Render the scene
        
        pygame.quit()
        sys.exit()


# Entry point
if __name__ == "__main__":
    game = Game(rendered=True)
    game.run()
