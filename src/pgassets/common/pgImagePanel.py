import pygame

from src.pgassets.common.pgObject import pgObject


class pgImagePanel(pgObject):
    def __init__(self, pos, size, img_path, color=(255, 255, 255), borderwidth=0, transparent=False, bordercolor=(0, 0, 0), id=0):
        pgObject.__init__(self, pos, size, color, borderwidth, transparent, bordercolor)
        self.id = id
        self.image = load_image(img_path)
        self.image_rect = self.image.get_rect()
        self.image_rect.center = self.rect.center

    def set_image(self, img_path):
        self.image = load_image(img_path)
        self.image_rect = self.image.get_rect()
        self.image_rect.center = self.rect.center

    def set_pos(self, pos):
        self.rect.left = pos[0]
        self.rect.top = pos[1]
        self.image_rect.center = self.rect.center

    def draw(self, screen: pygame.Surface):
        pgObject.draw(self, screen)
        screen.blit(self.image, self.image_rect)


def load_image(name: str) -> pygame.Surface:
    path = "images/"
    return pygame.image.load(path + name)
