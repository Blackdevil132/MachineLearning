import pygame


class pgObject:
    def __init__(self, pos: tuple, size: tuple, color: tuple=(0, 0, 0), borderwidth: int=0, transparent: bool=False, bordercolor: tuple=(0, 0, 0)):
        #print(pos, size)
        self.rect = pygame.Rect(*pos, *size)
        self.color = color
        self.borderwidth = borderwidth
        self.transparent = transparent
        self.bordercolor = bordercolor

    def collidepoint(self, pos: tuple) -> bool:
        return self.rect.collidepoint(*pos)

    def draw(self, screen: pygame.Surface):
        if not self.transparent:
            pygame.draw.rect(screen, self.color, self.rect)
        if self.borderwidth:
            pygame.draw.rect(screen, self.bordercolor, self.rect, self.borderwidth)