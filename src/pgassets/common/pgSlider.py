import numpy as np
import pygame

from src.pgassets import pgObject


class pgSlider(pgObject):
    def __init__(self, pos, size):
        pgObject.__init__(self, pos, size)
        self.slider_rect = pygame.Rect(pos, (size[0] * 0.8, 4))
        self.slider_rect.center = self.rect.center
        self.slider_button = self.slider_rect.center
        self.slider_button_radius = 8

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.slider_rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)
        pygame.draw.circle(screen, (255, 50, 50), self.slider_button, self.slider_button_radius)

    def collidepoint(self, pos):
        distance = np.hypot(pos[0] - self.slider_button[0], pos[1] - self.slider_button[1])

        if distance <= self.slider_button_radius:
            return True

    def update_slider(self, pos):
        if self.slider_rect.left <= pos <= self.slider_rect.right:
            self.slider_button = (pos, self.slider_button[1])

    def get_value(self):
        value = self.slider_button[0] - self.slider_rect.left
        return round(2*value/self.slider_rect.width, 1)