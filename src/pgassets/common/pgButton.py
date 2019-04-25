from src.pgassets import pgTextPanel


class pgButton(pgTextPanel):
    def __init__(self, pos: tuple, size: tuple, text="", color=(255, 255, 255), borderwidth=2, transparent=False, fontsize=20):
        pgTextPanel.__init__(self, pos, size, text, color, borderwidth, transparent, fontsize=fontsize)
