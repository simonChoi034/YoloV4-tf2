import numpy as np


# anchors
class Anchors:
    def __init__(self, image_size: int):
        self.yolo_anchors = np.array(
            [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
            np.float32)
        self.yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        self.image_size = image_size

    def set_image_size(self, image_size: int):
        self.image_size = image_size

    def get_anchors(self) -> np.ndarray:
        return self.yolo_anchors / self.image_size

    def get_anchor_masks(self) -> np.ndarray:
        return self.yolo_anchor_masks
