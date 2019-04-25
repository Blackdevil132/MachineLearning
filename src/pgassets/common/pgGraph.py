import pygame

from src.pgassets import pgObject


class pgGraph(pgObject):
    def __init__(self, pos, size):
        pgObject.__init__(self, pos, size)
        self.data = []
        self.datapoints = []
        self.graph_color = (255, 0, 0)
        self.graph_rect = pygame.Rect(*pos, *(size[0]*0.8, size[1]*0.7))
        self.graph_rect.center = self.rect.center

    def set_data(self, data):
        self.data = data[-self.graph_rect.width:]

        self.datapoints = []
        for i in range(len(self.data)):
            pos_y = int(self.data[i] * self.graph_rect.height)
            self.datapoints.append((self.graph_rect.left + i, self.graph_rect.bottom - pos_y))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)

        # draw points
        for point in self.datapoints:
                screen.fill(self.graph_color, (point, (1, 1)))

        # draw axes
        pygame.draw.line(screen, self.color, self.graph_rect.topleft, self.graph_rect.bottomleft, 2)
        pygame.draw.line(screen, self.color, self.graph_rect.bottomleft, self.graph_rect.bottomright, 2)

