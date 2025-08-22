import torch
import torch.nn as nn

# Import các module cần thiết
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
        W:    (B,T,K)
        """
        barX, W = self.pre(x)
        # barX = barX[:, :, self.joins_idx, :]
        # 1. Joint embedding
        E0 = self.joint_embed(barX, W)  # (B,T,K,d)
        # 2. Spatial encoder
        Es = self.spatial_enc(E0, barX, W)  # (B,T,K,d)
        # 3. Frame summary
        S = self.frame_summary(Es, W)  # (B,T,d)
        # 4. Temporal bottleneck (sequence -> Z hoặc S')
        Z, S_out = self.temp_bottleneck(S)  # (B,M,d), (B,T,d)
        # 5. Motion residual branch
        R = self.motion_branch(x, W) + S_out  # (B,T,d)
        # 6. Cross-fusion memory-motion
        Z_star, F = self.cross_fusion(Z, R)  # (B,M,d), (B,T,d)
        # 7. Classification head
        logits = self.head(F, Z_star)  # (B,num_classes)
        return logits


if __name__ == "__main__":
    import yaml
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Now import the modules
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

    # Load config from LSA-64.yaml
    config_path = "./configs/LSA-64.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please make sure the config file exists in the configs directory.")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=== ISLR Model Test with LSA-64 Config ===")
    print(f"Config loaded from: {config_path}")

    # Extract model config
    model_cfg = cfg["model"]

    # Test parameters
    batch_size = 2
    sequence_length = 64  # max_sequence_length from config
    d = model_cfg["d"]
    num_classes = model_cfg["num_classes"]

    # Extract parts and index from config
    parts = model_cfg["parts"]
    index = model_cfg["index"]

    print("\nModel Configuration:")
    print(f"  d (hidden_dim): {d}")
    print(f"  num_classes: {num_classes}")
    print(f"  parts: {list(parts.keys())}")
    print(f"  index: {index}")

    # Create test data
    num_joints = sum(len(v) for v in parts.values())
    barX = torch.randn(batch_size, sequence_length, num_joints, 2)
    W = torch.ones(batch_size, sequence_length, num_joints)

    print("\nTest Data Shapes:")
    print(f"  barX: {barX.shape}")
    print(f"  W: {W.shape}")

    # Create model with config
    try:
        model = JASTNet(**model_cfg)

        print(f"\nModel created successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        with torch.no_grad():
            logits = model(barX)
            print(f"Output logits shape: {logits.shape}")
            print(f"Expected shape: ({batch_size}, {num_classes})")

            # Test single batch
            barX_single = torch.randn(1, sequence_length, num_joints, 2)
            W_single = torch.ones(1, sequence_length, num_joints)
            logits_single = model(barX_single)
            print(f"Single batch output shape: {logits_single.shape}")

            print("\n✅ Forward pass test passed!")

    except Exception as e:
        print(f"\n❌ Error during model creation or forward pass: {e}")
        import traceback

        traceback.print_exc()

        print("\nDebug info:")
        print(f"  model_cfg keys: {list(model_cfg.keys())}")
        print(f"  parts: {parts}")
        print(f"  index: {index}")

    print("\n=== Test completed ===")
