import math
from typing import List, Dict

import numpy as np
import tensorflow as tf


class Yolov4DatasetGenerator:
    def __init__(self, image_input_size: List, anchors: np.ndarray, anchor_masks: np.ndarray):
        self.image_input_size = image_input_size
        self.anchors = anchors
        self.anchor_masks = anchor_masks

    def transform_targets_for_output(self, y_true: np.ndarray, grid_size: int, anchor_idxs: np.ndarray) -> np.ndarray:
        # y_true: (boxes, (x, y, w, h, class, best_anchor))
        # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
        y_true_out = np.zeros((grid_size, grid_size, anchor_idxs.shape[0], 6))

        for i in range(y_true.shape[0]):
            anchor_eq = np.equal(
                anchor_idxs, y_true[i][5]
            )

            if np.any(anchor_eq):
                box = y_true[i][0:4]
                box_xy = y_true[i][0:2]

                anchor_idx = np.where(anchor_eq)
                grid_xy = box_xy // (1 / grid_size)
                grid_xy = grid_xy.astype(int)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                y_true_out[grid_xy[1]][grid_xy[0]][anchor_idx[0][0]] = [box[0], box[1], box[2], box[3], 1, y_true[i][4]]

        return y_true_out

    def transform_label(self, y_true: np.ndarray) -> List[np.ndarray]:
        # y_train = [[x,y,w,h,c],...] shape=(n, 5)
        y_outs = []
        grid_size = math.ceil(self.image_input_size[0] / 32)

        anchor_area = self.anchors[..., 0] * self.anchors[..., 1]
        box_wh = y_true[..., 2:4]
        box_wh = np.tile(np.expand_dims(box_wh, -2), (1, 1, self.anchors.shape[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = np.minimum(box_wh[..., 0], self.anchors[..., 0]) * np.minimum(box_wh[..., 1],
                                                                                     self.anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = np.argmax(iou, axis=-1)
        anchor_idx = np.reshape(anchor_idx, (-1, 1))

        y_train = np.concatenate([y_true, anchor_idx], axis=-1)

        for anchor_idxs in self.anchor_masks:
            y_outs.append(self.transform_targets_for_output(y_train, grid_size, anchor_idxs))
            grid_size *= 2

        return y_outs

    def resize_label(self, label: np.ndarray, original_dim: List) -> np.ndarray:
        # change top-left xy to center xy
        # [x, y, w, h] -> [center_x, center_y, w, h]

        # normalize label
        img_h, img_w = original_dim
        target_h, target_w = self.image_input_size
        ratio_w = min(target_w / img_w, target_h / img_h) / target_w
        ratio_h = min(target_w / img_w, target_h / img_h) / target_h

        index = label.shape[0]

        multiplier = np.asarray([[ratio_w, ratio_h, ratio_w, ratio_h] for _ in range(index)])

        return label * multiplier

    def set_dataset_info(self):
        # setup the info of dataset for training
        pass

    def get_bbox(self, index: List[int]) -> np.ndarray:
        pass

    def gen_next_pair(self):
        pass


class YoloV4Dataset:
    def __init__(self, generator: Yolov4DatasetGenerator, image_input_size: List, batch_size: int, buffer_size: int, prefetch_size: int):
        self.generator = generator
        self.image_input_size = image_input_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size

    def resize_image(self, img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, self.image_input_size, preserve_aspect_ratio=True)
        img = tf.image.pad_to_bounding_box(img, 0, 0, self.image_input_size[0], self.image_input_size[1])
        return img / 127.5 - 1  # normalize to [-1, 1]

    def read_and_resize_image(self, element: Dict) -> Dict:
        img = tf.io.read_file(element['image'])
        img = tf.image.decode_jpeg(img)
        img.set_shape([None, None, 3])
        scale_1_label, scale_2_label, scale_3_label = element['scale_1_label'], element['scale_2_label'], element[
            'scale_3_label']

        # resize and pad image to required input size
        img = self.resize_image(img)
        element['image'] = img
        # format label
        element['label'] = tuple([scale_1_label, scale_2_label, scale_3_label])

        return element

    def create_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self.generator.gen_next_pair,
            output_types={
                'image': tf.string,
                'scale_1_label': tf.float32,
                'scale_2_label': tf.float32,
                'scale_3_label': tf.float32,
                'label_index': tf.int32
            }
        )
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(map_func=self.read_and_resize_image)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)

        return dataset
