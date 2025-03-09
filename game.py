import pygame
import sys
import math

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ROBOT_COLOR = (100, 100, 100)
WALL_THICKNESS = 20
TARGET_COLOR = (255, 240, 200)  # Light yellow/gold for target area

# Define target area (bottom right corner)
TARGET_SIZE = 150
TARGET_X = WIDTH - WALL_THICKNESS - TARGET_SIZE
TARGET_Y = HEIGHT - WALL_THICKNESS - TARGET_SIZE

# Define constants for completion detection
FRAMES_FOR_COMPLETION = 60  # Number of frames all balls must remain in target to be "done"

class PhysicsObject:
    """Base class for objects with physics"""
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vel_x = 0
        self.vel_y = 0
        self.mass = radius * radius  # Mass proportional to size
        self.friction = 0.01  # Base friction value

    def update(self, dt):
        # Apply friction
        self.vel_x *= (1 - self.friction)
        self.vel_y *= (1 - self.friction)
        
        # Basic physics update
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self._handle_wall_collision()

    def _handle_wall_collision(self):
        # Bounce off walls
        if self.x - self.radius < WALL_THICKNESS:
            self.x = WALL_THICKNESS + self.radius
            self.vel_x = -self.vel_x * 0.8  # Damping factor
        elif self.x + self.radius > WIDTH - WALL_THICKNESS:
            self.x = WIDTH - WALL_THICKNESS - self.radius
            self.vel_x = -self.vel_x * 0.8
            
        if self.y - self.radius < WALL_THICKNESS:
            self.y = WALL_THICKNESS + self.radius
            self.vel_y = -self.vel_y * 0.8
        elif self.y + self.radius > HEIGHT - WALL_THICKNESS:
            self.y = HEIGHT - WALL_THICKNESS - self.radius
            self.vel_y = -self.vel_y * 0.8

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, other):
        # Distance between centers
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if colliding
        if distance < self.radius + other.radius:
            return True
        return False

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

    def is_in_target_area(self):
        """Check if object is in the target area (bottom right corner)"""
        return (self.x > TARGET_X and self.y > TARGET_Y)

class Robot(PhysicsObject):
    def __init__(self, x, y):
        super().__init__(x, y, 25, ROBOT_COLOR)
        self.rotation = 0  # In radians
        self.motor_speeds = [0, 0]  # Left and right motor speeds
        self.angular_velocity = 0
        self.speed_multiplier = 0.5  # Reduce speed (was 1.0 implicitly before)
        self.friction = 0.05  # Robot has more friction than balls

    def set_action(self, action):
        # Update motor speeds based on action
        # Clamp values between -1 and 1
        self.motor_speeds[0] = max(-1, min(1, action[0])) * 100  # Scale to original range
        self.motor_speeds[1] = max(-1, min(1, action[1])) * 100  # Scale to original range
    
    def calculate_speed(self):
        # Average of the two motor speeds determines forward speed
        # Apply speed multiplier to make robot slower
        return (self.motor_speeds[0] + self.motor_speeds[1]) * self.speed_multiplier
    
    def calculate_angular_speed(self):
        # Difference between motor speeds determines rotation
        # Also scale down for smoother control
        return (self.motor_speeds[1] - self.motor_speeds[0]) * 0.08
    
    def update(self, dt):
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
        super().update(dt)
    
    def draw(self, screen):
        # Draw the robot body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw a line to show the orientation
        line_end_x = self.x + math.cos(self.rotation) * self.radius
        line_end_y = self.y + math.sin(self.rotation) * self.radius
        pygame.draw.line(screen, BLACK, (int(self.x), int(self.y)), 
                         (int(line_end_x), int(line_end_y)), 3)

class Ball(PhysicsObject):
    def __init__(self, x, y, color, radius=15):
        super().__init__(x, y, radius, color)
        self.friction = 0.03  # Increased friction for balls
        
        # Balls don't start with random velocity now - they'll be arranged in a pool formation

def create_pool_formation(center_x, center_y, ball_radius=15):
    """Create 3 balls in a triangular pool-like formation"""
    # Define the distance between ball centers
    spacing = ball_radius * 2.2  # Slightly more than 2 radii for a small gap
    
    # Create ball positions
    # First ball at the top of the triangle
    ball1_pos = (center_x, center_y - spacing/2)
    
    # Second and third balls form the base of the triangle
    ball2_pos = (center_x - spacing/2, center_y + spacing/2)
    ball3_pos = (center_x + spacing/2, center_y + spacing/2)
    
    return [
        Ball(ball1_pos[0], ball1_pos[1], RED),
        Ball(ball2_pos[0], ball2_pos[1], GREEN),
        Ball(ball3_pos[0], ball3_pos[1], BLUE)
    ]

class Game:
    def __init__(self, rendered=True, physics_steps=1):
        self.rendered = rendered
        self.physics_steps = physics_steps
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize pygame window only if in rendered mode
        if self.rendered:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Robot and Balls Physics Simulation")
        
        # Create robot away from the balls
        self.robot = Robot(WIDTH // 4, HEIGHT // 2)
        
        # Create balls in a pool-like formation in the right half of the screen
        self.balls = create_pool_formation(WIDTH * 3 // 4, HEIGHT // 2)
        
        # List to keep track of removed balls
        self.removed_balls = []
        
        # List of all physics objects for collision detection
        self.all_objects = [self.robot] + self.balls
        
        # Track the frame time for consistent physics in both modes
        self.dt = 1.0 / FPS
        
        # Track the previous positions for distance-based reward
        self.prev_ball_positions = [(ball.x, ball.y) for ball in self.balls]
        
        # Track total reward for episode
        self.total_reward = 0
        
        # Track whether the task is complete
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
        
        # Display the reward if in rendered mode
        if self.rendered:
            self.total_reward += reward
    
    def reset_simulation(self):
        # Reset robot position
        self.robot.x = WIDTH // 4
        self.robot.y = HEIGHT // 2
        self.robot.vel_x = 0
        self.robot.vel_y = 0
        self.robot.rotation = 0
        
        # Recreate balls in formation
        self.balls = create_pool_formation(WIDTH * 3 // 4, HEIGHT // 2)
        
        # Reset removed balls list
        self.removed_balls = []
        
        # Update object list
        self.all_objects = [self.robot] + self.balls
        
        # Reset tracking variables
        self.prev_ball_positions = [(ball.x, ball.y) for ball in self.balls]
        self.total_reward = 0
        self.done = False

    def calculate_reward(self):
        """
        Calculate reward based on the current state.
        
        The reward function includes:
        1. One-time reward of 100 for each ball when it enters the target area
        2. Reward for balls moving toward target (prev_dist - curr_dist)
        3. Small time penalty to encourage efficiency
        
        Returns:
            float: The calculated reward
        """
        reward = 0
        
        # Small time penalty to encourage efficiency
        reward -= 0.01
        
        # Target center coordinates
        target_center_x = TARGET_X + TARGET_SIZE/2
        target_center_y = TARGET_Y + TARGET_SIZE/2
        
        # Check each ball's position relative to target
        balls_to_remove = []
        
        for i, ball in enumerate(self.balls):
            # Skip balls that are already removed
            if ball in self.removed_balls:
                continue
                
            # Check if ball is in target area
            in_target = ball.is_in_target_area()
            
            if in_target:
                # Give a one-time reward for getting the ball in the target area
                reward += 100
                balls_to_remove.append(ball)
            else:
                # Distance-based reward component for balls not yet in target
                current_dist = math.sqrt((ball.x - target_center_x)**2 + (ball.y - target_center_y)**2)
                prev_dist = math.sqrt((self.prev_ball_positions[i][0] - target_center_x)**2 + 
                                    (self.prev_ball_positions[i][1] - target_center_y)**2)
                
                # Reward for moving toward target (raw distance difference)
                reward += (prev_dist - current_dist)
        
        # Update previous positions for next calculation (only for balls still in play)
        self.prev_ball_positions = [(ball.x, ball.y) for ball in self.balls if ball not in self.removed_balls]
        
        # Remove balls that entered the target area
        for ball in balls_to_remove:
            if ball in self.balls and ball not in self.removed_balls:
                self.balls.remove(ball)
                self.removed_balls.append(ball)
                # Update all_objects list as well
                if ball in self.all_objects:
                    self.all_objects.remove(ball)
        
        return reward
    
    def check_completion(self):
        """Check if the task is complete (all balls in target)"""
        # Episode is done when all balls have been removed (placed in target)
        return len(self.balls) == len(self.removed_balls) or all(ball in self.removed_balls for ball in self.balls)
    
    def action(self, motor_left, motor_right):
        """
        Apply motor commands and step the simulation.
        
        Args:
            motor_left (float): Left motor speed in range [-1, 1]
            motor_right (float): Right motor speed in range [-1, 1]
            
        Returns:
            tuple: (positions, reward, done)
                positions: list of coordinates [robot_x, robot_y, ball1_x, ball1_y, ...]
                reward: float value indicating the reward for this action
                done: boolean indicating if the episode is complete
        """
        # Set robot action
        self.robot.set_action([motor_left, motor_right])

        total_reward = 0
        
        # Run multiple physcics steps per action
        for _ in range(self.physics_steps):
            # Update physics (use fixed time step for consistent physics)
            if self.rendered:
                # In rendered mode, get the actual time delta
                self.dt = self.clock.tick(FPS) / 1000.0
            # else use the fixed dt defined in __init__
            
            # Update all objects
            for obj in self.all_objects:
                obj.update(self.dt)
            
            # Check for collisions between all objects
            for i in range(len(self.all_objects)):
                for j in range(i + 1, len(self.all_objects)):
                    if self.all_objects[i].check_collision(self.all_objects[j]):
                        self.all_objects[i].resolve_collision(self.all_objects[j])
            
            # Calculate reward
            step_reward = self.calculate_reward()
            total_reward += step_reward
            
            # Check if episode is done
            if not self.done:  # Only check if not already done
                self.done = self.check_completion()
                
                # Give a completion reward if we just finished
                if self.done:
                    total_reward += 50.0  # Big reward for maintaining all balls in target
                    break
        
        # Return state, reward, and done flag
        return self.get_state(), total_reward, self.done
    
    def get_state(self):
        """
        Return state representation:
        - angle_to_corner (scaled to [-1, 1])
        - angle_to_ball1, angle_to_ball2, angle_to_ball3 (scaled to [-1, 1])
        - robot x,y position (scaled to [-1, 1])
        - robot orientation (scaled to [-1, 1])
        """
        state = []
        
        # Target corner center coordinates
        target_center_x = TARGET_X + TARGET_SIZE/2
        target_center_y = TARGET_Y + TARGET_SIZE/2
        
        # Calculate angle to corner
        dx_corner = target_center_x - self.robot.x
        dy_corner = target_center_y - self.robot.y
        angle_to_corner = math.atan2(dy_corner, dx_corner)
        # Scale to [-1, 1]
        angle_to_corner_scaled = angle_to_corner / math.pi
        state.append(angle_to_corner_scaled)
        
        # Calculate angles to each ball
        all_balls = self.balls + self.removed_balls  # Consider all balls, including removed ones
        for ball in all_balls:
            if ball in self.removed_balls:
                # If ball is removed, use a default value to indicate it's in the target
                state.append(0)  # Neutral angle value when ball is removed
            else:
                dx_ball = ball.x - self.robot.x
                dy_ball = ball.y - self.robot.y
                angle_to_ball = math.atan2(dy_ball, dx_ball)
                # Scale to [-1, 1]
                angle_to_ball_scaled = angle_to_ball / math.pi
                state.append(angle_to_ball_scaled)
        
        # Add robot position scaled to [-1, 1]
        # Scale x from [WALL_THICKNESS, WIDTH-WALL_THICKNESS] to [-1, 1]
        x_scaled = 2 * (self.robot.x - WALL_THICKNESS) / (WIDTH - 2 * WALL_THICKNESS) - 1
        # Scale y from [WALL_THICKNESS, HEIGHT-WALL_THICKNESS] to [-1, 1]
        y_scaled = 2 * (self.robot.y - WALL_THICKNESS) / (HEIGHT - 2 * WALL_THICKNESS) - 1
        state.append(x_scaled)
        state.append(y_scaled)
        
        # Add robot orientation scaled to [-1, 1]
        orientation_scaled = self.robot.rotation / math.pi
        state.append(orientation_scaled)
        
        return state
    
    def draw(self):
        if not self.rendered:
            return
            
        self.screen.fill(WHITE)
        
        # Draw target area (bottom right corner)
        pygame.draw.rect(self.screen, TARGET_COLOR, 
                        (TARGET_X, TARGET_Y, TARGET_SIZE, TARGET_SIZE))
        
        # Draw a border for the target area
        pygame.draw.rect(self.screen, BLACK, 
                        (TARGET_X, TARGET_Y, TARGET_SIZE, TARGET_SIZE), 2)
        
        # Draw walls
        pygame.draw.rect(self.screen, BLACK, (0, 0, WIDTH, WALL_THICKNESS))
        pygame.draw.rect(self.screen, BLACK, (0, 0, WALL_THICKNESS, HEIGHT))
        pygame.draw.rect(self.screen, BLACK, (0, HEIGHT - WALL_THICKNESS, WIDTH, WALL_THICKNESS))
        pygame.draw.rect(self.screen, BLACK, (WIDTH - WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
        
        # Draw all objects
        for obj in self.all_objects:
            obj.draw(self.screen)
        
        # Draw instructions
        font = pygame.font.SysFont(None, 24)
        controls_text = "Controls: W,A,S,D to move | R to reset"
        text_surface = font.render(controls_text, True, BLACK)
        self.screen.blit(text_surface, (20, HEIGHT - 40))
        
        # Draw current reward
        reward_text = f"Total Reward: {self.total_reward:.2f}"
        reward_surface = font.render(reward_text, True, BLACK)
        self.screen.blit(reward_surface, (20, HEIGHT - 70))
        
        # Draw status of balls in target
        status_text = f"Balls in target: {len(self.removed_balls)}/{len(self.balls) + len(self.removed_balls)}"
        status_surface = font.render(status_text, True, BLACK)
        self.screen.blit(status_surface, (20, HEIGHT - 100))
        
        # Draw completion status
        if self.done:
            completion_text = "TASK COMPLETE!"
            completion_surface = font.render(completion_text, True, GREEN)
            text_rect = completion_surface.get_rect(center=(WIDTH//2, 50))
            self.screen.blit(completion_surface, text_rect)
        
        pygame.display.flip()
    
    def update(self):
        """Wrapper for action() that uses keyboard controls"""
        # This method is maintained for backward compatibility
        # For rendered mode, process events and handle keyboard input
        if self.rendered:
            self.process_events()
    
    def run(self):
        """Run the game loop (only used in rendered mode)"""
        if not self.rendered:
            return
            
        while self.running:
            self.update()  # Process input and update physics
            self.draw()    # Render the scene
        
        pygame.quit()
        sys.exit()
