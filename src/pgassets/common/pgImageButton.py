from src.pgassets import pgImagePanel


class pgImageButton(pgImagePanel):
    def __init__(self, pos, size, img_path, color=(255, 255, 255), borderwidth=2, transparent=False):
        pgImagePanel.__init__(self, pos, size, img_path, color, borderwidth=borderwidth, transparent=transparent)
