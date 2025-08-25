import torch
import torch.nn as nn

try:
    from models.layers import (
        ClassificationHead,
        CrossFusionMemoryMotion,
        MotionResidualBranch,
    )
    from models.preprocessor import Preprocessor
    from models.jointEmbedding import JointEmbedding
    from models.spatialEncoder import SpatialEncoder
    from models.frameSummary import FrameSummary
    from models.temporalBottleneck import TemporalBottleneck
except ImportError:
    from layers import (
        ClassificationHead,
        CrossFusionMemoryMotion,
        MotionResidualBranch,
    )
    from preprocessor import Preprocessor
    from jointEmbedding import JointEmbedding
    from spatialEncoder import SpatialEncoder
    from frameSummary import FrameSummary
    from temporalBottleneck import TemporalBottleneck


class JASTNet(nn.Module):
    def __init__(
        self,
        d: int,
        num_classes: int,
        parts: dict,
        index: dict,
        heads: int = 4,
        ff_dim: int = 128,
        **cfg,
    ):
        super().__init__()
        self.pre = Preprocessor(
            index=index,
            parts=parts,
            **cfg.get("preprocessor", {}),
        )
        num_joints = sum(len(v) for v in parts.values())
        self.keypoints_index = parts.copy()

        self.parts = parts
        self.joint_embed = JointEmbedding(
            K=num_joints, d=d, part_to_indices=parts, **cfg.get("joint_embedding", {})
        )
        self.spatial_enc = SpatialEncoder(
            d=d, heads=heads, d_ff=ff_dim, **cfg.get("spatial_encoder", {})
        )
        self.frame_summary = FrameSummary(
            d=d,
            parts_order=list(parts.keys()),
            part_to_indices=parts,
            **cfg.get("frame_summary", {}),
        )
        self.temp_bottleneck = TemporalBottleneck(
            d=d,
            heads=heads,
            d_ff=ff_dim,
            **cfg.get("temp_bottleneck", {}),
        )
        self.motion_branch = MotionResidualBranch(
            d=d, parts=parts, **cfg.get("motion_branch", {})
        )
        self.cross_fusion = CrossFusionMemoryMotion(
            d=d,
            heads=heads,
            **cfg.get("cross_fusion", {}),
        )
        self.head = ClassificationHead(
            d=d,
            num_classes=num_classes,
            **cfg.get("classification_head", {}),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B,T,K,2)
        """
        barX, W = self.pre(x)
        E0 = self.joint_embed(barX, W)  # (B,T,K,d)
        Es = self.spatial_enc(E0, barX, W)  # (B,T,K,d)
        S = self.frame_summary(Es, W)  # (B,T,d)
        Z, S_out = self.temp_bottleneck(S)  # (B,M,d), (B,T,d)
        R = self.motion_branch(x, W) + S_out  # (B,T,d)
        Z_star, F = self.cross_fusion(Z, R)  # (B,M,d), (B,T,d)
        logits = self.head(F, Z_star)  # (B,num_classes)
        return logits


if __name__ == "__main__":
    import yaml

    config_path = "./configs/ASL-Citizen-200.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]

    batch_size = 2
    sequence_length = 64
    d = model_cfg["d"]
    num_classes = model_cfg["num_classes"]

    parts = model_cfg["parts"]
    index = model_cfg["index"]

    num_joints = sum(len(v) for v in parts.values())
    x = torch.randn(batch_size, sequence_length, num_joints, 2)

    model = JASTNet(**model_cfg)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters:,}")
    logits = model(x)
    print(logits.shape)
