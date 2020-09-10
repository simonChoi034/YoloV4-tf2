import argparse
import colorsys
import datetime
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from config import cfg
from dataset.coco_classes import coco_classes
from dataset.yolov4_coco_dataset import COCO2017Dataset
from metrics.mean_average_precision.detection_map import DetectionMAP
from model.loss import YOLOv4Loss
from model.utils import non_max_suppression
from model.yolov4 import YOLOv4
from utils.lr_schedule import WarmUpLinearCosineDecay

try:
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


class Trainer:
    def __init__(self, batch_size: int, image_size: int):
        # setup anchors
        cfg.anchors.set_image_size(image_size)

        # dataset
        dataset_train = COCO2017Dataset(image_size=image_size, batch_size=batch_size)
        dataset_val = COCO2017Dataset(mode=tfds.Split.VALIDATION, image_size=image_size,
                                      batch_size=batch_size)
        self.dataset_train = dataset_train.get_dataset()
        self.dataset_val = dataset_val.get_dataset()

        # parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.buffer_size = cfg.buffer_size
        self.prefetch_size = cfg.prefetch_size
        self.num_class = cfg.num_class
        self.yolo_iou_threshold = cfg.yolo_iou_threshold
        self.yolo_score_threshold = cfg.yolo_score_threshold
        self.label_smoothing_factor = cfg.label_smoothing_factor
        self.lr_init = cfg.lr_init
        self.lr_end = cfg.lr_end
        self.warmup_epochs = cfg.warmup_epochs
        self.train_epochs = cfg.train_epochs
        self.warmup_steps = self.warmup_epochs * dataset_train.num_of_img / self.batch_size
        self.total_steps = self.train_epochs * dataset_train.num_of_img / self.batch_size
        self.step_to_log = cfg.step_to_log

        # define model and loss
        self.model = YOLOv4(num_class=self.num_class)
        self.lr_scheduler = WarmUpLinearCosineDecay(warmup_steps=self.warmup_steps, decay_steps=self.total_steps, initial_learning_rate=self.lr_init)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler)
        self.checkpoint_dir = './checkpoints/yolov4_train.tf'
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=5)
        self.loss_fn = YOLOv4Loss(num_class=self.num_class, yolo_iou_threshold=self.yolo_iou_threshold,
                                  label_smoothing_factor=self.label_smoothing_factor, use_ciou_loss=True)

        # metrics
        self.mAP = DetectionMAP(self.num_class)

        # summary writer
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/yolov4/train'
        self.val_log_dir = 'logs/yolov4/val'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

    def plot_bounding_box(self, images: tf.Tensor, bboxes, scores, class_ids, valid_detections):
        image = images.numpy()[0]
        valid_detection = valid_detections.numpy()[0]
        bbox = bboxes.numpy()[0][:valid_detection]
        score = scores.numpy()[0][:valid_detection]
        class_id = class_ids.numpy()[0][:valid_detection]

        hsv_tuples = [(1.0 * x / self.num_class, 1., 1.) for x in range(self.num_class)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        fontScale = 0.5

        image_h, image_w, _ = image.shape
        image = (image + 1) / 2 * 255
        image = image.astype(np.uint8)

        for i, (box, sc, cls) in enumerate(zip(bbox, score, class_id)):
            x1, y1, x2, y2 = box
            x1 = int(x1 * image_w)
            x2 = int(x2 * image_w)
            y1 = int(y1 * image_h)
            y2 = int(y2 * image_h)

            cls = int(cls)
            bbox_thick = int(0.6 * (image_h + image_w) / self.image_size)
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], bbox_thick)

            bbox_mess = '%s: %.2f' % (coco_classes[cls], sc)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), colors[cls], -1)  # filled

            cv2.putText(image, bbox_mess, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        return np.expand_dims(image, 0)

    @tf.function
    def validation(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # calculate loss from validation dataset
        pred = self.model(x)
        pred_loss = self.loss_fn(y_pred=pred, y_true=y)

        # get bounding box
        bboxes, scores, classes, valid_detections = non_max_suppression(pred)

        return pred_loss, bboxes, scores, classes, valid_detections

    @tf.function
    def train_one_step(self, x: tf.Tensor, y: tf.Tensor) -> List[tf.Tensor]:
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            pred_loss = self.loss_fn(y_pred=pred, y_true=y)
            regularization_loss = tf.reduce_sum(self.model.losses)
            total_loss = pred_loss + regularization_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return pred_loss

    def log_metrics(self, writer: tf.summary.SummaryWriter, dataset: tf.data.Dataset):
        batch_loss = tf.Variable(0.0, dtype=tf.float32)
        for _ in range(self.batch_size):
            data = next(iter(dataset))
            loss, bboxes, scores, class_ids, valid_detections = self.validation(data['image'], data['label'])

            gt_boxes = data["bbox"]
            num_of_gt_boxes = data["num_of_bbox"]

            batch_loss = batch_loss + loss

            # calculate mAP
            for frame in zip(bboxes.numpy(), class_ids.numpy(), scores.numpy(), valid_detections.numpy(),
                             gt_boxes.numpy(),
                             num_of_gt_boxes.numpy()):
                pred_bbox, pred_cls, pred_score, valid_detection, gt_box, num_of_gt_box = frame

                # get all predicion and label
                pred_bbox = pred_bbox[:valid_detection]
                pred_cls = pred_cls[:valid_detection]
                pred_score = pred_score[:valid_detection]
                gt_box = gt_box[:num_of_gt_box]
                gt_bbox = gt_box[..., :4]
                gt_class_id = gt_box[..., 4]

                #
                frame = pred_bbox, pred_cls, pred_score, gt_bbox, gt_class_id
                self.mAP.evaluate(*frame)

        mean_average_precision = self.mAP.get_mAP()
        self.mAP.reset_accumulators()

        # plot image
        pred_image = self.plot_bounding_box(data['image'], bboxes, scores, class_ids, valid_detections)
        gt_image = self.plot_bounding_box(data['image'], gt_boxes[..., :4], tf.ones_like(scores), gt_boxes[..., 4],
                                          num_of_gt_boxes)

        # log tensorboard
        step = int(self.ckpt.step)
        with writer.as_default():
            tf.summary.scalar("lr", self.optimizer.lr(step), step=step)
            tf.summary.scalar('loss', batch_loss, step=step)
            tf.summary.scalar('mean loss', batch_loss.numpy() / self.batch_size,
                              step=step)
            tf.summary.scalar('mAP@0.5', mean_average_precision, step=step)
            tf.summary.image("Display pred bounding box", pred_image, step=step)
            tf.summary.image("Display gt bounding box", gt_image, step=step)

    def train_one_epoch(self):
        for data in self.dataset_train:
            loss = self.train_one_step(data['image'], data['label'])

            # validation every i steps
            if int(self.ckpt.step) % self.step_to_log == 0:
                self.log_metrics(self.train_summary_writer, self.dataset_train)
                self.log_metrics(self.val_summary_writer, self.dataset_val)

                # Save checkpoint
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def train(self):
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for e in range(self.train_epochs):
            self.train_one_epoch()

    def main(self):
        self.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('-b', '--batch_size', type=int, default=cfg.batch_size, help='Batch size')
    parser.add_argument('-i', '--image_size', type=int, default=cfg.image_size, help='Reshape size of the image')
    args = parser.parse_args()

    trainer = Trainer(
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    trainer.main()
