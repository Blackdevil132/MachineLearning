import pygame

from src.pgassets.common.pgObject import pgObject


class pgGrid(pgObject):
    def __init__(self, pos, size, shape, objects, color=(0, 0, 0), borderwidth=0, transparent=True, bordercolor=(0, 0, 0)):
        pgObject.__init__(self, pos, size, color, borderwidth, transparent, bordercolor)
        self.shape = shape
        self.objects = objects

        # calculate grid
        h_offset, v_offset = 64, 64
        gap_x = (self.rect.width - self.shape[1]*self.objects[0].rect.width - 2*h_offset) / (self.shape[1] - 1)
        gap_y = (self.rect.height - self.shape[0]*self.objects[0].rect.height - 2*v_offset) / (self.shape[0] - 1)

        for j in range(self.shape[0]):
            for i in range(self.shape[1]):
                index = self.shape[1]*j + i
                self.objects[index].set_pos((int(self.rect.left + h_offset + i*gap_x + i*self.objects[index].rect.width),
                                            int(self.rect.top + v_offset + j*gap_y + j*self.objects[index].rect.height)))

    def collidepoint(self, pos: tuple):
        for o in self.objects:
            if o.collidepoint(pos):
                return o.id
        return 0

    def draw(self, screen: pygame.Surface):
        pgObject.draw(self, screen)
        for o in self.objects:
            o.draw(screen)
