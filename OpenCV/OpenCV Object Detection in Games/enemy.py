class Enemy:
    def __init__(self, name: str, image_path, x_left, x_right, y_up, y_bottom):
        self.name = name
        self.image_path = image_path
        self.x_left = x_left
        self.x_right = x_right
        self.y_up = y_up
        self.y_bottom = y_bottom

class Bonus:
    def __init__(self, name: str, x, y, w, h):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h