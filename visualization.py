import pygame
import threading
import time

class NetworkVisualizer:
    def __init__(self, actor):
        self.actor = actor
        self.update_interval = 5  # Update every 5 seconds
        self.node_radius = 20
        self.node_color = (0, 0, 255)
        self.edge_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.width, self.height = 800, 600
        self.offset_x, self.offset_y = 0, 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.screen = None

    def build_graph(self):
        self.nodes = []
        self.edges = []
        layers = [self.actor.fc1, self.actor.fc2, self.actor.fc3]
        x_spacing = self.width // (len(layers) + 1)
        
        for i, layer in enumerate(layers):
            y_spacing = self.height // (layer.in_features + 1)
            for j in range(layer.in_features):
                self.nodes.append((x_spacing * i + self.offset_x, y_spacing * (j + 1) + self.offset_y))
            y_spacing = self.height // (layer.out_features + 1)
            for k in range(layer.out_features):
                self.nodes.append((x_spacing * (i + 1) + self.offset_x, y_spacing * (k + 1) + self.offset_y))
                for j in range(layer.in_features):
                    self.edges.append((x_spacing * i + self.offset_x, y_spacing * (j + 1) + self.offset_y,
                                       x_spacing * (i + 1) + self.offset_x, y_spacing * (k + 1) + self.offset_y))

    def draw_graph(self):
        self.screen.fill(self.bg_color)
        for edge in self.edges:
            pygame.draw.line(self.screen, self.edge_color, edge[:2], edge[2:], 1)
        for node in self.nodes:
            pygame.draw.circle(self.screen, self.node_color, node, self.node_radius)
        pygame.display.flip()

    def update_graph(self):
        while True:
            self.build_graph()
            self.draw_graph()
            time.sleep(self.update_interval)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.drag_start = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx, dy = event.rel
                    self.offset_x += dx
                    self.offset_y += dy

    def start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Neural Network Visualization")
        thread = threading.Thread(target=self.update_graph)
        thread.daemon = True
        thread.start()
        while True:
            self.handle_events()
            time.sleep(0.01)
