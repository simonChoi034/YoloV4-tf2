from easydict import EasyDict

from model.anchor import Anchors

cfg = EasyDict()

cfg.dataset = "coco/2017"
cfg.num_of_img = 118287
cfg.num_class = 80
cfg.grid_sensitivity_ratio = 1.1
cfg.yolo_score_threshold = 0.3
cfg.yolo_iou_threshold = 0.45
cfg.label_smoothing_factor = 0.1
cfg.image_size = 608
cfg.buffer_size = 100
cfg.batch_size = 4
cfg.prefetch_size = 5
cfg.lr_init = 1e-3
cfg.lr_end = 1e-6
cfg.warmup_epochs = 2
cfg.train_epochs = 300
cfg.step_to_log = 250
cfg.anchors = Anchors(cfg.image_size)
