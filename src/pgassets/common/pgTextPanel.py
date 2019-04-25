import pygame

from src.pgassets.common.pgObject import pgObject


class pgTextPanel(pgObject):
    def __init__(self, pos, size, text, color=(255, 255, 255), borderwidth=0, transparent=False, bordercolor=(0, 0, 0), textcolor=(0, 0, 0), fontsize=20, bold=False):
        pgObject.__init__(self, pos, size, color, borderwidth, transparent, bordercolor)
        self.font = pygame.font.SysFont("arial", fontsize)
        self.font.set_bold(bold)
        self.textcolor = textcolor
        self.text = text

        self.label = self.font.render(text, 1, self.textcolor)
        self.label_rect = self.label.get_rect()
        self.label_rect.center = self.rect.center

    def set_text(self, text: str):
        self.text = text
        self.label = self.font.render(text, 1, self.textcolor)
        self.label_rect = self.label.get_rect()
        self.label_rect.center = self.rect.center

    def set_pos(self, pos):
        self.rect.left = pos[0]
        self.rect.top = pos[1]

    def draw(self, screen):
        pgObject.draw(self, screen)
        screen.blit(self.label, self.label_rect)