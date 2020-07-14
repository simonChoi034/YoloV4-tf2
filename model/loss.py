from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss, binary_crossentropy

from config import cfg
from model.utils import decode


class YOLOv4Loss(Loss):
    def __init__(
            self,
            num_class: int,
            yolo_iou_threshold: float,
            label_smoothing_factor: float = 0,
            use_focal_loss: bool = False,
            use_focal_obj_loss: bool = False,
            use_giou_loss: bool = False,
            use_ciou_loss: bool = False):
        super(YOLOv4Loss, self).__init__()
        self.num_class = num_class
        self.yolo_iou_threshold = yolo_iou_threshold
        self.label_smoothing_factor = label_smoothing_factor
        self.use_focal_obj_loss = use_focal_obj_loss
        self.use_focal_loss = use_focal_loss
        self.use_giou_loss = use_giou_loss
        self.use_ciou_loss = use_ciou_loss
        self.anchors = cfg.anchors.get_anchors()
        self.anchor_masks = cfg.anchors.get_anchor_masks()

    @staticmethod
    def smooth_labels(y_true: tf.Tensor, smoothing_factor: Union[tf.Tensor, float],
                      num_class: Union[tf.Tensor, int] = 1) -> tf.Tensor:
        return y_true * (1.0 - smoothing_factor) + smoothing_factor / num_class

    @staticmethod
    def broadcast_iou(box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                           tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                           tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                     (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                     (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    @staticmethod
    def iou(box_1: tf.Tensor, box_2: tf.Tensor) -> tf.Tensor:
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (..., (x1, y1, x2, y2))
        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                           tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                           tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                     (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                     (box_2[..., 3] - box_2[..., 1])

        iou = int_area / (box_1_area + box_2_area - int_area)
        return iou

    @staticmethod
    def giou(box_1: tf.Tensor, box_2: tf.Tensor) -> tf.Tensor:
        # box_1: (batch_size, grid_y, grid_x, N, (x1, y1, x2, y2))
        # box_2: (batch_size, grid_y, grid_x, N, (x1, y1, x2, y2))
        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                           tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                           tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                     (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                     (box_2[..., 3] - box_2[..., 1])
        union_area = box_1_area + box_2_area - int_area
        iou = int_area / union_area

        enclose_left_up = tf.minimum(box_1[..., :2], box_2[..., :2])
        enclose_right_down = tf.maximum(box_1[..., 2:], box_2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    @staticmethod
    def ciou(box_1: tf.Tensor, box_2: tf.Tensor) -> tf.Tensor:
        # box_1: (batch_size, grid_y, grid_x, N, (x1, y1, x2, y2))
        # box_2: (batch_size, grid_y, grid_x, N, (x1, y1, x2, y2))
        box_1 = tf.concat([tf.minimum(box_1[..., :2], box_1[..., 2:]),
                           tf.maximum(box_1[..., :2], box_1[..., 2:])], axis=-1)
        box_2 = tf.concat([tf.minimum(box_2[..., :2], box_2[..., 2:]),
                           tf.maximum(box_2[..., :2], box_2[..., 2:])], axis=-1)

        # box area
        box_1_w, box_1_h = box_1[..., 2] - box_1[..., 0], box_1[..., 3] - box_1[..., 1]
        box_2_w, box_2_h = box_2[..., 2] - box_2[..., 0], box_2[..., 3] - box_2[..., 1]
        box_1_area = box_1_w * box_1_h
        box_2_area = box_2_w * box_2_h

        # find iou
        left_up = tf.maximum(box_1[..., :2], box_2[..., :2])
        right_down = tf.minimum(box_1[..., 2:], box_2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = box_1_area + box_2_area - inter_area
        iou = inter_area / (union_area + K.epsilon())

        # find enclosed area
        enclose_left_up = tf.minimum(box_1[..., :2], box_2[..., :2])
        enclose_right_down = tf.maximum(box_1[..., 2:], box_2[..., 2:])

        enclose_wh = enclose_right_down - enclose_left_up
        enclose_c2 = tf.square(enclose_wh[..., 0]) + tf.square(enclose_wh[..., 1])

        box_1_center_x = (box_1[..., 0] + box_1[..., 2]) / 2
        box_1_center_y = (box_1[..., 1] + box_1[..., 3]) / 2
        box_2_center_x = (box_2[..., 0] + box_2[..., 2]) / 2
        box_2_center_y = (box_2[..., 1] + box_2[..., 3]) / 2

        p2 = tf.square(box_1_center_x - box_2_center_x) + tf.square(box_1_center_y - box_2_center_y)

        # and epsilon to prevent nan
        atan1 = tf.atan(box_1_w / (box_1_h + K.epsilon()))
        atan2 = tf.atan(box_2_w / (box_2_h + K.epsilon()))
        v = 4.0 * tf.square(atan1 - atan2) / (np.pi ** 2)
        a = v / (1 - iou + v + K.epsilon())

        ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
        return ciou

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, gamma: Union[tf.Tensor, float] = 2.0,
                   alpha: Union[tf.Tensor, float] = 0.25, label_smoothing: Union[tf.Tensor, float] = 0) -> tf.Tensor:
        sigmoid_loss = binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
        sigmoid_loss = tf.expand_dims(sigmoid_loss, axis=-1)

        p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

        sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss

        sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

        return sigmoid_focal_loss

    def loss_layer(self, y_pred: tf.Tensor, y_true: tf.Tensor, anchors: np.array) -> tf.Tensor:
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...class))
        # pred_box_coor: (batch_size, grid, grid, anchors, (x1, y1, x2, y2))
        pred_box_coor, pred_obj, pred_class = decode(y_pred, anchors)

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...class))
        true_box, true_obj, true_class = tf.split(
            y_true, (4, 1, self.num_class), axis=-1)
        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        # (batch_size, grid, grid, anchors, (x1, y1, x2, y2))
        true_box_coor = tf.concat([true_xy - true_wh / 2, true_xy + true_wh / 2], axis=-1)

        # smooth label
        true_class = YOLOv4Loss.smooth_labels(true_class, smoothing_factor=self.label_smoothing_factor,
                                              num_class=self.num_class)

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou, _, _ = tf.map_fn(
            lambda x: (tf.reduce_max(YOLOv4Loss.broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1), 0, 0),
            (pred_box_coor, true_box_coor, obj_mask))

        ignore_mask = tf.cast(best_iou < self.yolo_iou_threshold, tf.float32)

        # 4. calculate all losses
        # confidence loss
        if self.use_focal_obj_loss:
            confidence_loss = self.focal_loss(true_obj, pred_obj)
        else:
            confidence_loss = binary_crossentropy(true_obj, pred_obj)
            confidence_loss = obj_mask * confidence_loss + (1 - obj_mask) * ignore_mask * confidence_loss

        # class loss
        if self.use_focal_loss:
            class_loss = self.focal_loss(true_class, pred_class)
        else:
            class_loss = obj_mask * binary_crossentropy(true_class, pred_class)

        # box loss
        if self.use_giou_loss:
            giou = self.giou(pred_box_coor, true_box_coor)
            box_loss = obj_mask * box_loss_scale * (1 - giou)
        elif self.use_ciou_loss:
            ciou = self.ciou(pred_box_coor, true_box_coor)
            box_loss = obj_mask * box_loss_scale * (1 - ciou)

        # sum of all loss
        box_loss = tf.reduce_sum(box_loss, axis=(1, 2, 3))
        confidence_loss = tf.reduce_sum(confidence_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return box_loss + confidence_loss + class_loss

    def yolo_loss(self, pred_sbbox: tf.Tensor, pred_mbbox: tf.Tensor, pred_lbbox: tf.Tensor, true_sbbox: tf.Tensor,
                  true_mbbox: tf.Tensor, true_lbbox: tf.Tensor) -> tf.Tensor:
        loss_sbbox = self.loss_layer(pred_sbbox, true_sbbox, self.anchors[self.anchor_masks[0]])
        loss_mbbox = self.loss_layer(pred_mbbox, true_mbbox, self.anchors[self.anchor_masks[1]])
        loss_lbbox = self.loss_layer(pred_lbbox, true_lbbox, self.anchors[self.anchor_masks[2]])

        return tf.reduce_sum(loss_sbbox + loss_mbbox + loss_lbbox)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_s, true_m, true_l = y_true
        pred_s, pred_m, pred_l = y_pred
        loss = self.yolo_loss(pred_s, pred_m, pred_l, true_s, true_m, true_l)

        return loss
