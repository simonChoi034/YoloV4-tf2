import math
from typing import Dict, Tuple, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

from config import cfg


class COCO2017Dataset:
    def __init__(
            self,
            dataset: str = cfg.dataset,
            num_class: int = cfg.num_class,
            mode: Any = tfds.Split.TRAIN,
            image_size: int = cfg.image_size,
            batch_size: int = cfg.batch_size,
            buffer_size: int = cfg.buffer_size,
            prefetch_size: int = cfg.prefetch_size):
        self.dataset = tfds.load(name=dataset, split=mode)
        self.num_of_img = cfg.num_of_img
        self.num_class = num_class
        self.image_size = image_size  # [height, width]
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.max_bbox_size = 100
        self.anchors = cfg.anchors.get_anchors()
        self.anchor_masks = cfg.anchors.get_anchor_masks()

    def map_func(self, feature: tf.Tensor) -> Dict:
        image = feature["image"]
        bbox = feature["objects"]["bbox"]
        num_of_bbox = tf.shape(bbox)[0]

        original_image_size = tf.shape(image)[0:2]
        bbox = tf.numpy_function(self.transform_bbox, inp=[bbox, original_image_size], Tout=tf.float32)

        label_small, label_medium, label_large = tf.numpy_function(self.map_label_func,
                                                                   inp=[bbox, feature["objects"]["label"]],
                                                                   Tout=[tf.float32, tf.float32, tf.float32])
        image = self.map_image_func(image)

        bbox = self.pad_class(bbox, feature["objects"]["label"])
        bbox = self.pad_bbox(bbox)

        feature_dict = {
            "image": image,
            "label": (label_small, label_medium, label_large),
            "bbox": bbox,
            "num_of_bbox": num_of_bbox
        }

        return feature_dict

    def map_label_func(self, bbox: np.ndarray, label: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        # bbox = [x_min, y_min, x_max, y_max] => [x_center, y_center, w, h]
        bbox_min = bbox[..., 0:2]
        bbox_max = bbox[..., 2:4]
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_wh = bbox_max - bbox_min
        bbox = np.concatenate([bbox_center, bbox_wh], axis=-1)

        # clip value
        bbox = np.clip(bbox, a_min=0.0, a_max=1 - K.epsilon())

        # convert to yolo label format
        # bbox = [[x,y,w,h],...] shape=(1, n, 4)
        grid_size = math.ceil(self.image_size / 32)

        # find the best anchor
        anchor_area = self.anchors[..., 0] * self.anchors[..., 1]
        box_wh = bbox[..., 2:4]
        box_wh = np.tile(np.expand_dims(box_wh, -2), (1, 1, self.anchors.shape[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = np.minimum(box_wh[..., 0], self.anchors[..., 0]) * np.minimum(box_wh[..., 1],
                                                                                     self.anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = np.argmax(iou, axis=-1)  # shape = (1, n)

        label_small = self.create_yolo_label(bbox, label, self.anchor_masks[0], anchor_idx, grid_size)
        label_medium = self.create_yolo_label(bbox, label, self.anchor_masks[1], anchor_idx, grid_size * 2)
        label_large = self.create_yolo_label(bbox, label, self.anchor_masks[2], anchor_idx, grid_size * 4)

        return label_small, label_medium, label_large

    def map_image_func(self, image: np.ndarray) -> tf.Tensor:
        img = tf.image.resize(image, (self.image_size, self.image_size), preserve_aspect_ratio=True)
        img = tf.image.pad_to_bounding_box(img, 0, 0, self.image_size, self.image_size)
        img = tf.image.random_brightness(img, max_delta=0.25)
        img = tf.image.random_contrast(img, lower=0.4, upper=1.3)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_saturation(img, lower=0, upper=4)

        img = img / 127.5 - 1  # normalize to [-1, 1]

        return img

    def create_yolo_label(self, bbox: np.ndarray, label: np.ndarray, anchor_mask: np.ndarray, anchor_idx: np.ndarray,
                          grid_size: int) -> np.ndarray:
        # bbox.shape = (n, 4)
        # label.shape = (n)
        # anchor_idx.shape = (n)
        # girds.shape = (grid_size, grid_size, B, (5 + N))
        grids = np.zeros((grid_size, grid_size, anchor_mask.shape[0], (5 + self.num_class)), dtype=np.float32)

        # put
        for box, class_id, anchor_id in zip(bbox, label, anchor_idx):
            if anchor_id in anchor_mask:
                box = box[0:4]
                box_xy = box[0:2]

                grid_xy = box_xy // (1 / grid_size)
                grid_xy = grid_xy.astype(int)

                box_index = np.where(anchor_mask == anchor_id)[0][0]

                grid_array = np.zeros((5 + self.num_class))
                grid_array[0:5] = np.array([box[0], box[1], box[2], box[3], 1])
                class_index = int(5 + class_id)
                grid_array[class_index] = 1

                # grid[y][x][anchor] = [tx, ty, bw, bh, obj, ...class_id]
                grids[grid_xy[1]][grid_xy[0]][box_index] = grid_array

        return grids

    def transform_bbox(self, bbox: np.ndarray, original_image_size: np.ndarray) -> np.ndarray:
        # bbox = [y_min, x_min, y_max, x_max] => [x_min, y_min, x_max, y_max]
        # bbox.shape: (n, 4)
        bbox[..., 0:2] = bbox[..., 0:2][..., ::-1]
        bbox[..., 2:4] = bbox[..., 2:4][..., ::-1]

        # rescale bbox to fit new image size
        orig_img_h, orig_img_w = original_image_size[0], original_image_size[1]
        target_img_h, target_img_w = self.image_size, self.image_size
        ratio_w = min(target_img_w / orig_img_w, target_img_h / orig_img_h) * (orig_img_w / target_img_w)
        ratio_h = min(target_img_w / orig_img_w, target_img_h / orig_img_h) * (orig_img_h / target_img_h)

        multiplier = np.array([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)

        bbox = bbox * multiplier

        return bbox

    def pad_bbox(self, bbox: tf.Tensor) -> Tuple[tf.Tensor]:
        # bbox.shape = (n, 5)
        bbox = tf.expand_dims(bbox, axis=-1)  # bbox.shape = (n, 5, 1)
        bbox = tf.image.pad_to_bounding_box(bbox, 0, 0, self.max_bbox_size, tf.shape(bbox)[1])

        bbox = tf.squeeze(bbox)

        return bbox

    def pad_class(self, bbox: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        # bbox.shape = (n, 4)
        # label.shape = (n)
        label = tf.cast(tf.reshape(label, (-1, 1)), tf.float32)
        label = tf.concat([bbox, label], axis=-1)
        return label

    def get_dataset(self):
        dataset = self.dataset.filter(lambda x: tf.shape(x["objects"]["bbox"])[0] != 0) \
            .map(self.map_func) \
            .shuffle(self.buffer_size) \
            .batch(self.batch_size) \
            .prefetch(self.prefetch_size)

        return dataset
