"""IWNET architecture - the research core of this project.

RESEARCH METHODOLOGY (deliberately unchanged)
---------------------------------------------
    Input
      -> EfficientNet-B3 (frozen, stages 2/3/4 as features_only out_indices=(1,2,3))
      -> Channel Attention   (per-stage)
      -> Spatial Attention   (per-stage)
      -> IW Attention        (per-stage)
      -> 1x1 projection to D (per-stage)
      -> bilinear alignment of all stages to the deepest stage's spatial size
      -> adaptive scale weighting (softmax over 3 scales, driven by global
         pooled concatenated features)
      -> multi-scale fusion (weighted sum)
      -> classification head
      -> 7-class output

Nothing in that chain has been replaced, reordered or simplified. The head
widths, the attention formulas, the fusion arithmetic and the softmax scale
weighting are byte-for-byte the original design.

Engineering fixes applied here (none change the architecture):
  * Feature channel counts are read from the backbone at construction time, as
    before - there are no hard-coded channel numbers anywhere.
  * ``forward`` validates every intermediate shape and raises
    :class:`FeatureShapeError` naming the feature, the expected shape and the
    actual shape, instead of letting an opaque PyTorch broadcast error surface
    from deep inside the network.
  * ``pretrained`` is a parameter. The original always passed
    ``pretrained=True``, so *inference* re-downloaded EfficientNet weights even
    though the checkpoint's own ``state_dict`` was loaded immediately after -
    which breaks offline inference and wastes bandwidth.
  * **BatchNorm freezing fix.** ``requires_grad=False`` does not stop
    BatchNorm running statistics from updating: the modules stayed in ``train()``
    mode and EfficientNet-B3's ``running_mean``/``running_var`` drifted on a
    small dataset for the first 10 epochs. Frozen BatchNorm is now pinned to
    ``eval()`` for as long as the backbone is frozen. This is a correctness fix,
    not a methodology change - the intent of the original code was clearly a
    frozen backbone.
  * ``IWAttention`` is documented as producing a *per-sample scalar gate*
    (see :class:`IWAttention`). It is preserved exactly as written; its limited
    expressive power is flagged in the README rather than silently "fixed".
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from iwnet.config import ARCHITECTURE, BACKBONE, OUT_INDICES, Config, FINAL_CLASSES

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "IWAttention",
    "FeatureShapeError",
    "IWNET",
    "build_model",
    "describe_model",
]


class FeatureShapeError(RuntimeError):
    """Raised when a backbone feature map does not match the expected contract."""


# ─────────────────────────────────────────────────────────────────────────────
#  ATTENTION BLOCKS  (unchanged from the original design)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """Squeeze-and-excite style channel re-weighting via global average pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.fc(x.mean(dim=[2, 3])))
        return x * weights[:, :, None, None]


class SpatialAttention(nn.Module):
    """Spatial re-weighting from the channel-mean and channel-max maps."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class IWAttention(nn.Module):
    """Channel + Spatial + a learned per-sample gate.

    The gate is ``AdaptiveAvgPool2d(1) -> Flatten -> Linear(C, 1) -> Sigmoid``,
    i.e. the entire feature map collapses to ONE scalar per image. The gate can
    therefore only scale a sample uniformly - it carries no channel or spatial
    information. This is preserved verbatim from the original model; see README
    "Limitations" for why it is scientifically questionable.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.iw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x * self.iw(x).view(-1, 1, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
class IWNET(nn.Module):
    """EfficientNet-B3 + multi-stage IW attention + adaptive scale fusion."""

    def __init__(
        self,
        num_classes: int = len(FINAL_CLASSES),
        D: int = 128,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        dropout: float = 0.5,
        dropout_hidden: float = 0.3,
        backbone: str = BACKBONE,
        out_indices: Sequence[int] = OUT_INDICES,
        image_size: int = 224,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.num_classes = num_classes
        self.D = D
        self.image_size = image_size
        self.backbone_name = backbone
        self.out_indices = tuple(out_indices)
        self._backbone_frozen = False

        self.backbone = self._create_backbone(timm, backbone, self.out_indices, pretrained)

        # Channel counts come from the backbone, never from a constant.
        with torch.no_grad():
            probe = self.backbone(torch.zeros(1, 3, image_size, image_size))
        if len(probe) != len(self.out_indices):
            raise FeatureShapeError(
                f"Backbone '{backbone}' returned {len(probe)} feature maps for "
                f"out_indices={self.out_indices}; expected {len(self.out_indices)}."
            )
        self.feature_channels: tuple[int, ...] = tuple(int(f.shape[1]) for f in probe)
        C1, C2, C3 = self.feature_channels

        self.att1 = IWAttention(C1)
        self.att2 = IWAttention(C2)
        self.att3 = IWAttention(C3)

        self.proj1 = nn.Conv2d(C1, D, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(C2, D, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(C3, D, kernel_size=1, bias=False)

        self.scale_w = nn.Linear(D * 3, 3)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(D, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_hidden),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    # ── construction helpers ───────────────────────────────────────────────
    @staticmethod
    def _create_backbone(timm, name: str, out_indices, pretrained: bool) -> nn.Module:
        """Create the feature extractor, with an actionable message on failure."""
        try:
            return timm.create_model(
                name, pretrained=pretrained, features_only=True, out_indices=tuple(out_indices)
            )
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not create backbone '{name}' with pretrained weights: {exc}\n"
                    "Common causes:\n"
                    "  * no internet access and the weights are not in the local timm/HF cache\n"
                    "  * a corrupted partial download in the cache directory\n"
                    "Workarounds:\n"
                    "  * run once while online to populate the cache, or\n"
                    "  * pass --no-pretrained to train from scratch (changes results), or\n"
                    "  * set HF_HOME to a directory that already contains the weights\n"
                    "Re-run 'python grape.py --check-environment' for a full report."
                ) from exc
            raise

    # ── freezing ───────────────────────────────────────────────────────────
    def _set_backbone_frozen(self, frozen: bool) -> None:
        self._backbone_frozen = frozen
        for param in self.backbone.parameters():
            param.requires_grad = not frozen
        if frozen:
            self._pin_batchnorm_eval(self.backbone)

    @staticmethod
    def _pin_batchnorm_eval(module: nn.Module) -> None:
        """Force every BatchNorm inside ``module`` into eval mode.

        Without this, ``requires_grad=False`` leaves BatchNorm *statistics*
        updating: the pretrained backbone's running means/variances drift on a
        small dataset before it is ever trained.
        """
        for submodule in module.modules():
            if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
                submodule.eval()

    def freeze_backbone(self) -> None:
        self._set_backbone_frozen(True)

    def train(self, mode: bool = True) -> "IWNET":
        super().train(mode)
        if self._backbone_frozen:
            self._pin_batchnorm_eval(self.backbone)
        return self

    def unfreeze_backbone(self, last_n: int = 3) -> int:
        """Unfreeze the last ``last_n`` backbone stages. Returns how many."""
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            self._set_backbone_frozen(False)
            log_msg = "backbone fully unfrozen (no stage list available)"
            print(f"[INFO] {log_msg}.")
            return -1

        total = len(blocks)
        count = max(0, min(int(last_n), total))
        start = total - count
        for index in range(start, total):
            for param in blocks[index].parameters():
                param.requires_grad = True
        self._backbone_frozen = False
        print(f"[INFO] Unfroze last {count}/{total} backbone stages (blocks {start}..{total - 1}).")
        return count

    def backbone_stage_count(self) -> int:
        return len(getattr(self.backbone, "blocks", []) or [])

    # ── forward ────────────────────────────────────────────────────────────
    @staticmethod
    def _expect(name: str, tensor: torch.Tensor, channels: int, rank: int = 4) -> None:
        """Validate a feature map and raise a diagnostic error on mismatch."""
        if tensor.dim() != rank:
            raise FeatureShapeError(
                f"Feature '{name}': expected a {rank}D tensor (B, C, H, W), "
                f"got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[1] != channels:
            raise FeatureShapeError(
                f"Feature '{name}': expected channel count {channels}, "
                f"got {tensor.shape[1]} (full shape {tuple(tensor.shape)}). "
                f"The backbone output layout changed - re-derive the projections."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise FeatureShapeError(
                f"Input 'x': expected a 4D tensor (B, 3, H, W), got shape {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise FeatureShapeError(
                f"Input 'x': expected 3 channels (RGB), got {x.shape[1]} "
                f"(full shape {tuple(x.shape)})."
            )

        features = self.backbone(x)
        if len(features) != 3:
            raise FeatureShapeError(
                f"Backbone '{self.backbone_name}' returned {len(features)} feature maps "
                f"for out_indices={self.out_indices}; expected 3. "
                f"Shapes: {[tuple(f.shape) for f in features]}"
            )

        f1, f2, f3 = features
        C1, C2, C3 = self.feature_channels
        self._expect("stage1 (out_indices[0])", f1, C1)
        self._expect("stage2 (out_indices[1])", f2, C2)
        self._expect("stage3 (out_indices[2])", f3, C3)

        p1 = self.proj1(self.att1(f1))
        p2 = self.proj2(self.att2(f2))
        p3 = self.proj3(self.att3(f3))
        for name, tensor in (("proj1", p1), ("proj2", p2), ("proj3", p3)):
            if tensor.shape[1] != self.D:
                raise FeatureShapeError(
                    f"'{name}': expected {self.D} channels after 1x1 projection, "
                    f"got {tensor.shape[1]} (full shape {tuple(tensor.shape)})."
                )

        # Align every scale to the deepest stage's spatial resolution.
        target = p3.shape[-2:]
        p1_up = F.interpolate(p1, target, mode="bilinear", align_corners=False)
        p2_up = F.interpolate(p2, target, mode="bilinear", align_corners=False)

        # Adaptive scale weighting.
        global_features = torch.cat(
            [p1_up.mean([2, 3]), p2_up.mean([2, 3]), p3.mean([2, 3])], dim=1
        )
        expected_gate_in = self.D * 3
        if global_features.shape[1] != expected_gate_in:
            raise FeatureShapeError(
                f"'scale_weight_input': expected {expected_gate_in} features "
                f"(D*3 = {self.D}*3), got {global_features.shape[1]} "
                f"(full shape {tuple(global_features.shape)})."
            )
        weights = torch.softmax(self.scale_w(global_features), dim=1)
        if weights.shape != (x.shape[0], 3):
            raise FeatureShapeError(
                f"'scale_weights': expected shape {(x.shape[0], 3)}, "
                f"got {tuple(weights.shape)}."
            )

        fused = (
            weights[:, 0, None, None, None] * p1_up
            + weights[:, 1, None, None, None] * p2_up
            + weights[:, 2, None, None, None] * p3
        )
        return self.head(fused)

    # ── introspection ──────────────────────────────────────────────────────
    def describe(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": ARCHITECTURE,
            "backbone": self.backbone_name,
            "out_indices": list(self.out_indices),
            "feature_channels": list(self.feature_channels),
            "embedding_dim": self.D,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "backbone_frozen": self._backbone_frozen,
        }


def build_model(cfg: Config, *, pretrained: bool | None = None, num_classes: int | None = None) -> IWNET:
    """Construct :class:`IWNET` from configuration."""
    return IWNET(
        num_classes=num_classes if num_classes is not None else cfg.model.num_classes,
        D=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
        dropout=cfg.model.dropout,
        dropout_hidden=cfg.model.dropout_hidden,
        backbone=cfg.model.backbone,
        out_indices=cfg.model.out_indices,
        image_size=cfg.train.image_size,
        pretrained=cfg.model.pretrained if pretrained is None else pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    )


def describe_model(model: IWNET) -> str:
    info = model.describe()
    lines = [
        f"  Architecture      : {info['architecture']}",
        f"  Backbone          : {info['backbone']}  out_indices={info['out_indices']}",
        f"  Feature channels  : {info['feature_channels']}",
        f"  Embedding dim (D) : {info['embedding_dim']}",
        f"  Output classes    : {info['num_classes']}",
        f"  Image size        : {info['image_size']}",
        f"  Total parameters  : {info['total_parameters']:,}",
        f"  Trainable params  : {info['trainable_parameters']:,}"
        f"  ({info['trainable_parameters'] / max(info['total_parameters'], 1) * 100:.1f}%)",
        f"  Backbone frozen   : {info['backbone_frozen']}",
    ]
    return "\n".join(lines)
