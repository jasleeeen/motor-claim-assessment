"""Run the trained detectors on a photograph.
Weights are the two best.pt files produced by the Colab notebook:

    damage, 6 classes  (Faster R-CNN / ResNet-50 @640)
    parts, 21 classes  (Faster R-CNN / MobileNetV3 @640)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.vision.linking import Detection, SCORE_THRESHOLD, VisionResult, link, merge


class VisionUnavailable(RuntimeError):
    """torch is not installed, or the weights are missing."""


def _require_torch():
    try:
        import torch
        import torchvision 
    except (ImportError, OSError) as exc:
        raise VisionUnavailable(
            "torch and torchvision could not be loaded, so photographs cannot be "
            "read. Install them, or enter the damaged parts by hand.") from exc
    return torch


def _build(num_classes: int, min_sizes, max_size, backbone: str):
    from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                              fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    kw = {"min_size": tuple(min_sizes), "max_size": max_size,
          "weights": None, "weights_backbone": None}
    model = (fasterrcnn_resnet50_fpn_v2(**kw) if backbone == "resnet50"
             else fasterrcnn_mobilenet_v3_large_fpn(**kw))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def _infer_backbone(state: dict) -> str:
    """ResNet-50 state dicts carry backbone.body.layer1.*; MobileNetV3 does not."""
    return "resnet50" if any(k.startswith("backbone.body.layer1.") for k in state) \
        else "mobilenet"


@dataclass
class Detector:
    model: object
    classes: list[str]
    device: object

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None) -> "Detector":
        torch = _require_torch()
        path = Path(checkpoint)
        if not path.exists():
            raise VisionUnavailable(f"model weights not found: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = _build(len(ck["classes"]), ck["min_sizes"], ck["max_size"],
                       _infer_backbone(ck["model"]))
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        return cls(model, ck["classes"], dev)

    def detect(self, image) -> list[Detection]:
        torch = _require_torch()
        from torchvision.transforms import functional as TF

        with torch.no_grad():
            tensor = TF.to_tensor(image.convert("RGB")).to(self.device)
            out = self.model([tensor])[0]
        return [Detection(self.classes[int(l) - 1], float(s),
                          tuple(float(v) for v in b))
                for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(),
                                   out["scores"].cpu())]


class ClaimVision:
    """Both detectors together: photographs in, damaged parts out."""

    def __init__(self, damage_weights: str | Path, parts_weights: str | Path,
                 device: str | None = None,
                 score_threshold: float = SCORE_THRESHOLD):
        self.damage = Detector.load(damage_weights, device)
        self.parts = Detector.load(parts_weights, device)
        self.score_threshold = score_threshold

    def assess_photo(self, image) -> VisionResult:
        return link(self.damage.detect(image), self.parts.detect(image),
                    score_threshold=self.score_threshold)

    def assess_photos(self, images) -> VisionResult:
        return merge([self.assess_photo(im) for im in images])