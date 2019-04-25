import pygame


class pgCheckbox:
    def __init__(self, pos, size, color):
        self.rect = pygame.Rect(*pos, *size)
        self.color = color
        self.checked = False

    def collidepoint(self, pos):
        return self.rect.collidepoint(*pos)

    def update_status(self):
        self.checked = (self.checked + 1) % 2

    def get_status(self):
        return self.checked

    def draw(self, screen):
        if self.checked:
            pygame.draw.line(screen, self.color, self.rect.topleft, self.rect.bottomright, 5)
            pygame.draw.line(screen, self.color, self.rect.bottomleft, self.rect.topright, 5)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
