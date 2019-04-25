import pygame

from src.pgassets.common.pgImagePanel import pgImagePanel


types = {b'S': "grass_128.png", b'F': "grass_128.png", b'H': "grass_crackhole_128.png",
         b'G': "grass_treasure_128.png", b'HE': "grass_crackhole_enemy_128.png", b'FK': "grass_knight_128.png",
         b'FE': "grass_enemy_128.png", b'GE': "grass_treasure_enemy_128.png", b'GK': "grass_treasure_knight_128.png"}


class pgField(pgImagePanel):
    def __init__(self, pos, size, f_type, transparent=False, id=0):
        self.f_type = f_type
        pgImagePanel.__init__(self, pos, size, types[f_type], borderwidth=1, transparent=transparent, id=id)

    def set_type(self, f_type):
        self.f_type = f_type
        self.set_image(types[f_type])
