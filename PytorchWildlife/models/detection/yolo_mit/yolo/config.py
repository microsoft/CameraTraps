from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from torch import nn


@dataclass
class AnchorConfig:
    strides: List[int]
    reg_max: Optional[int]
    anchor_num: Optional[int]
    anchor: List[List[int]]


@dataclass
class LayerConfg:
    args: Dict
    source: Union[int, str, List[int]]
    tags: str


@dataclass
class BlockConfig:
    block: List[Dict[str, LayerConfg]]


@dataclass
class ModelConfig:
    name: Optional[str]
    anchor: AnchorConfig
    model: Dict[str, BlockConfig]


@dataclass
class DownloadDetail:
    url: str
    file_size: int


@dataclass
class DownloadOptions:
    details: Dict[str, DownloadDetail]


@dataclass
class DatasetConfig:
    path: str
    class_num: int
    class_list: List[str]
    auto_download: Optional[DownloadOptions]


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    cpu_num: int
    image_size: List[int]
    data_augment: Dict[str, int]
    source: Optional[Union[str, int]]
    dynamic_shape: Optional[bool]


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float
    momentum: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class MatcherConfig:
    iou: str
    topk: int
    factor: Dict[str, int]


@dataclass
class LossConfig:
    objective: Dict[str, int]
    aux: Union[bool, float]
    matcher: MatcherConfig


@dataclass
class SchedulerConfig:
    type: str
    warmup: Dict[str, Union[int, float]]
    args: Dict[str, Any]


@dataclass
class EMAConfig:
    enable: bool
    decay: float


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float
    max_bbox: int


@dataclass
class InferenceConfig:
    task: str
    nms: NMSConfig
    data: DataConfig
    fast_inference: Optional[None]
    save_predict: bool


@dataclass
class ValidationConfig:
    task: str
    nms: NMSConfig
    data: DataConfig


@dataclass
class TrainConfig:
    task: str
    epoch: int
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    ema: EMAConfig
    validation: ValidationConfig


@dataclass
class Config:
    task: Union[TrainConfig, InferenceConfig, ValidationConfig]
    dataset: DatasetConfig
    model: ModelConfig
    name: str

    device: Union[str, int, List[int]]
    cpu_num: int

    image_size: List[int]

    out_path: str
    exist_ok: bool

    lucky_number: 10
    use_wandb: bool
    use_tensorboard: bool

    weight: Optional[str]


@dataclass
class YOLOLayer(nn.Module):
    source: Union[int, str, List[int]]
    output: bool
    tags: str
    layer_type: str
    usable: bool
    external: Optional[dict]