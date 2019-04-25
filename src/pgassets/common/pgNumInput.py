import pygame

from src.pgassets import pgImageButton
from src.pgassets import pgObject
from src.pgassets import pgTextPanel


class pgNumInput(pgObject):
    def __init__(self, pos, size, default=2):
        pgObject.__init__(self, pos, size, transparent=True)
        self.value = default

        self.value_label = pgTextPanel((self.rect.left, self.rect.top + self.rect.height * 0.25),
                                       (self.rect.width, self.rect.height * 0.5),
                                       str(default), transparent=True, fontsize=30)

        self.inc_arrow = pgImageButton(self.rect.topleft, (self.rect.width, self.rect.height * 0.25), "play_icon_32.png",
                                       transparent=True, borderwidth=0)
        self.inc_arrow.image = pygame.transform.rotate(self.inc_arrow.image, 90)
        self.dec_arrow = pgImageButton((self.rect.left, self.rect.top + self.rect.height * 0.75), (self.rect.width, self.rect.height * 0.25),
                                       "play_icon_32.png", transparent=True, borderwidth=0)
        self.dec_arrow.image = pygame.transform.rotate(self.dec_arrow.image, -90)

    def inc_value(self):
        self.value += 1
        self.value_label.set_text(str(self.value))

    def dec_value(self):
        self.value -= 1
        self.value_label.set_text(str(self.value))

    def get_value(self):
        return self.value

    def collidepoint(self, pos: tuple):
        if self.inc_arrow.collidepoint(pos):
            self.inc_value()
            return 1
        elif self.dec_arrow.collidepoint(pos) and self.value > 1:
            self.dec_value()
            return 1
        return 0

    def draw(self, screen: pygame.Surface):
        self.value_label.draw(screen)
        self.inc_arrow.draw(screen)
        self.dec_arrow.draw(screen)
