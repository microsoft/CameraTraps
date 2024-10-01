__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import torch

import torch.nn.functional as F

from typing import Optional, List, Dict

from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,  _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor, GeneralizedRCNNTransform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads

from .register import MODELS

# adapted from torchvision implementation: https://github.com/pytorch/vision/blob/c890a7e75ebeaaa75ae9ace4c203b7fc145df068/torchvision/models/detection/roi_heads.py#L12
def fastrcnn_loss(class_logits, class_weights, box_regression, labels, regression_targets):
    '''
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        class_weights (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    '''

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels, weight=class_weights.to(labels.device))

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

# adapted from torchvision implementation: https://github.com/pytorch/vision/blob/c890a7e75ebeaaa75ae9ace4c203b7fc145df068/torchvision/models/detection/roi_heads.py#L492
class RoiHeadsWeightedLoss(RoIHeads):

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def forward(
        self,
        features,
        proposals,
        image_shapes, 
        targets=None, 
        ):
        '''
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        '''
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, self.class_weights, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses


@MODELS.register()
class FasterRCNNResNetFPN(GeneralizedRCNN):
    '''
    Build a Faster R-CNN model with a ResNet-FPN backbone. The ResNet architecture have to be specified in
    'architecture' argument. 

    This class was inspired by the ``fasterrcnn_resnet50_fpn`` function of torchvision, for details
    see https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L299.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Args:
        architecture (str): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool, optional): If True, returns a model with backbone pre-trained on Imagenet. 
            Defautls to True
        trainable_backbone_layers (int, optional): number of trainable (not frozen) resnet layers starting from 
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
            Defaults to None
        **kwargs: additional FasterRCNN arguments
    '''

    def __init__(
        self,
        architecture: str,
        num_classes: int, 
        pretrained_backbone: bool = True, 
        trainable_backbone_layers: Optional[int] = None, 
        anchor_sizes: Optional[tuple] = None,
        class_weights: Optional[list] = None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs
        ) -> None:

        assert architecture in ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)
        backbone = resnet_fpn_backbone(architecture, pretrained_backbone, trainable_layers=trainable_backbone_layers)
        
        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            if anchor_sizes is not None:
                anchor_sizes = tuple([tuple([i,]) for i in anchor_sizes])
                aspects = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
                rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspects)
            else:
                anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
                rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        if class_weights is not None:
            class_weights = torch.Tensor(class_weights)
            roi_heads = RoiHeadsWeightedLoss(
                class_weights=class_weights,
                box_roi_pool=box_roi_pool,
                box_head=box_head,
                box_predictor=box_predictor,
                fg_iou_thresh=box_fg_iou_thresh,
                bg_iou_thresh=box_bg_iou_thresh,
                batch_size_per_image=box_batch_size_per_image,
                positive_fraction=box_positive_fraction,
                bbox_reg_weights=bbox_reg_weights,
                score_thresh=box_score_thresh,
                nms_thresh=box_nms_thresh,
                detections_per_img=box_detections_per_img,
                )
        else:
            roi_heads = RoIHeads(
                box_roi_pool,
                box_head,
                box_predictor,
                box_fg_iou_thresh,
                box_bg_iou_thresh,
                box_batch_size_per_image,
                box_positive_fraction,
                bbox_reg_weights,
                box_score_thresh,
                box_nms_thresh,
                box_detections_per_img,
            )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)        