from typing import List, Optional, Union

import torch
from torch import Tensor, tensor
from torchvision.ops import batched_nms

from yolo.config import AnchorConfig, NMSConfig
from yolo.model.yolo import YOLO


def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        if torch.__version__ >= "2.3.0":
            anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        else:
            anchor_h, anchor_w = torch.meshgrid(h, w)
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


class Vec2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            self.strides = anchor_cfg.strides
        else:
            self.strides = self.create_auto_anchor(model, image_size)

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        dummy_input = torch.zeros(1, 3, H, W).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(W // anchor_num[1])
        return strides

    def update(self, image_size):
        """
        image_size: W, H
        """
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)

    def __call__(self, predicts):
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            B, C, h, w = pred_cls.shape
            pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(B, h * w, C)
            preds_cls.append(pred_cls)

            B, A, R, h, w = pred_anc.shape
            pred_anc = pred_anc.permute(0, 3, 4, 2, 1).reshape(B, h * w, R, A)
            preds_anc.append(pred_anc)

            B, X, h, w = pred_box.shape
            pred_box = pred_box.permute(0, 2, 3, 1).reshape(B, h * w, X)
            preds_box.append(pred_box)
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box


class Anc2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            self.strides = anchor_cfg.strides
        else:
            self.strides = self.create_auto_anchor(model, image_size)

        self.head_num = len(anchor_cfg.anchor)
        self.anchor_grids = self.generate_anchors(image_size)
        self.anchor_scale = tensor(anchor_cfg.anchor, device=device).view(self.head_num, 1, -1, 1, 1, 2)
        self.anchor_num = self.anchor_scale.size(2)
        self.class_num = model.num_classes

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        dummy_input = torch.zeros(1, 3, H, W).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head.shape
            strides.append(W // anchor_num[1])
        return strides

    def generate_anchors(self, image_size: List[int]):
        anchor_grids = []
        for stride in self.strides:
            W, H = image_size[0] // stride, image_size[1] // stride
            anchor_h, anchor_w = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
            anchor_grid = torch.stack((anchor_w, anchor_h), 2).view((1, 1, H, W, 2)).float().to(self.device)
            anchor_grids.append(anchor_grid)
        return anchor_grids

    def update(self, image_size):
        self.anchor_grids = self.generate_anchors(image_size)

    def __call__(self, predicts: List[Tensor]):
        preds_box, preds_cls, preds_cnf = [], [], []
        for layer_idx, predict in enumerate(predicts):
            B, LC, h, w = predict.shape
            L = self.anchor_num
            C = LC // L
            predict = predict.view(B, L, C, h, w).permute(0, 1, 3, 4, 2)  # B, L, h, w, C

            pred_box, pred_cnf, pred_cls = predict.split((4, 1, self.class_num), dim=-1)
            pred_box = pred_box.sigmoid()

            pred_box[..., 0:2] = (
                (pred_box[..., 0:2] * 2.0 - 0.5 + self.anchor_grids[layer_idx]) * self.strides[layer_idx]
            )
            pred_box[..., 2:4] = (
                (pred_box[..., 2:4] * 2) ** 2 * self.anchor_scale[layer_idx]
            )

            B, L, h, w, A = pred_box.shape
            preds_box.append(pred_box.reshape(B, L * h * w, A))

            B, L, h, w, C = pred_cls.shape
            preds_cls.append(pred_cls.reshape(B, L * h * w, C))

            preds_cnf.append(pred_cnf.reshape(B, L * h * w, C))

        preds_box = torch.concat(preds_box, dim=1)
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cnf = torch.concat(preds_cnf, dim=1)

        preds_box = transform_bbox(preds_box, "xycwh -> xyxy")
        return preds_cls, None, preds_box, preds_cnf.sigmoid()


def create_converter(model_version: str = "v9-c", *args, **kwargs) -> Union[Anc2Box, Vec2Box]:
    if "v7" in model_version:  # check model if v7
        converter = Anc2Box(*args, **kwargs)
    else:
        converter = Vec2Box(*args, **kwargs)
    return converter


def bbox_nms(cls_dist: Tensor, bbox: Tensor, nms_cfg: NMSConfig, confidence: Optional[Tensor] = None):
    cls_dist = cls_dist.sigmoid() * (1 if confidence is None else confidence)

    batch_idx, valid_grid, valid_cls = torch.where(cls_dist > nms_cfg.min_confidence)
    valid_con = cls_dist[batch_idx, valid_grid, valid_cls]
    valid_box = bbox[batch_idx, valid_grid]

    nms_idx = batched_nms(valid_box, valid_con, batch_idx + valid_cls * bbox.size(0), nms_cfg.min_iou)
    predicts_nms = []
    for idx in range(cls_dist.size(0)):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]]

        predict_nms = torch.cat(
            [valid_cls[instance_idx][:, None], valid_box[instance_idx], valid_con[instance_idx][:, None]], dim=-1
        )

        predicts_nms.append(predict_nms[: nms_cfg.max_bbox])
    return predicts_nms
