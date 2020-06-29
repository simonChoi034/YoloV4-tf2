import numpy as np
from dataset.yolov4_dataset import Yolov4DatasetGenerator


class MAFAGenerator(Yolov4DatasetGenerator):
    def __init__(self, dataset_dir, label_dir, image_input_size, anchors, anchor_masks):
        super().__init__(image_input_size, anchors, anchor_masks)
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks
