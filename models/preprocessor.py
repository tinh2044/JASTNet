from typing import Dict, List, Tuple
import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Preprocessor(nn.Module):
    def __init__(
        self,
        index: dict,
        parts: Dict[str, List[int]],
        eps: float = 1e-6,
        conf_alpha: float = 1.0,
        **cfg,
    ):
        super().__init__()
        self.body_index = index["body"]
        self.left_index = index["left"]
        self.right_index = index["right"]
        self.parts = parts
        self.eps = eps
        self.conf_alpha = conf_alpha

    def generate_confidence_weights(self, X: torch.Tensor) -> torch.Tensor:
        """
        Tự động sinh ra confidence weights dựa trên input coordinates
        Args:
            X: Input tensor of shape (B,T,K,2) or (T,K,2)
        Returns:
            W: Generated confidence weights of shape (B,T,K) or (T,K)
        """
        # Tính confidence dựa trên khoảng cách từ root joint
        if X.dim() == 3:
            X = X.unsqueeze(0)
            has_batch = False
        else:
            has_batch = True

        B, T, K, _ = X.shape
        X_flat = X.view(B * T, K, 2)

        # Lấy root joint (giả sử là joint đầu tiên)
        root_pos = X_flat[:, 0:1, :]  # (B*T, 1, 2)

        # Tính khoảng cách từ mỗi joint đến root
        distances = torch.norm(X_flat - root_pos, dim=2)  # (B*T, K)

        # Chuyển đổi khoảng cách thành confidence (khoảng cách càng nhỏ, confidence càng cao)
        # Sử dụng exponential decay
        max_dist = distances.max(dim=1, keepdim=True)[0]  # (B*T, 1)
        normalized_dist = distances / (max_dist + self.eps)

        # Confidence = exp(-distance), càng xa root càng thấp confidence
        confidence = torch.exp(
            -normalized_dist * 2.0
        )  # Scale factor 2.0 để điều chỉnh độ dốc

        # Áp dụng gamma function
        W = self.gamma(confidence)  # (B*T, K)

        # Reshape về batch format
        W = W.view(B, T, K)

        # Nếu input ban đầu không có batch, trả về không có batch
        if not has_batch:
            W = W.squeeze(0)

        return W

    def gamma(self, c: torch.Tensor) -> torch.Tensor:
        if self.conf_alpha == 1.0:
            return c.clamp(0.0, 1.0)
        return c.clamp(0.0, 1.0) ** self.conf_alpha

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: Input tensor of shape (B,T,K,2) or (T,K,2) - joint coordinates

        Returns:
            X_scaled: Preprocessed coordinates of shape (B,T,K,2) or (T,K,2)
            W: Generated confidence weights of shape (B,T,K) or (T,K)
        """
        # X: (B,T,K,2) hoặc (T,K,2)
        if X.dim() == 3:
            # Không có batch size, thêm batch dimension
            X = X.unsqueeze(0)  # (1,T,K,2)
            has_batch = False
        else:
            has_batch = True

        assert X.dim() == 4 and X.size(3) == 2, "X must be (B,T,K,2) or (T,K,2)"
        B, T, K, _ = X.shape

        # Reshape để xử lý batch
        X_flat = X.view(B * T, K, 2)  # (B*T, K, 2)

        # Xử lý từng frame trong batch
        r_t = X_flat[:, self.body_index, :].unsqueeze(1)  # (B*T,1,2)
        X_centered = X_flat - r_t  # (B*T,K,2)

        left = X_centered[:, self.left_index, :]  # (B*T,2)
        right = X_centered[:, self.right_index, :]  # (B*T,2)
        v = left - right  # (B*T,2)
        norms = torch.norm(v, dim=1, keepdim=True)  # (B*T,1)
        u = v / (norms + self.eps)  # (B*T,2)

        ux = u[:, 0].unsqueeze(1)  # (B*T,1)
        uy = u[:, 1].unsqueeze(1)  # (B*T,1)
        R = torch.stack(
            [torch.cat([ux, uy], dim=1), torch.cat([-uy, ux], dim=1)], dim=1
        )
        small_mask = norms.squeeze(1) <= self.eps
        if small_mask.any():
            R[small_mask] = torch.eye(2, dtype=X.dtype, device=X.device)

        X_rot = torch.einsum("tkd, tdc -> tkc", X_centered, R)  # (B*T,K,2)
        X_scaled = X_rot.clone()
        for part_name, idxs in self.parts.items():
            if len(idxs) == 0:
                continue
            sel = X_rot[:, idxs, :]  # (B*T, P, 2)
            rms = torch.sqrt((sel.pow(2).sum(dim=2).mean(dim=1)) + self.eps)  # (B*T,)
            rms_exp = rms.view(B * T, 1, 1)  # (B*T,1,1)
            X_scaled[:, idxs, :] = sel / rms_exp  # normalized per-part per-frame

        # Tự động sinh ra confidence weights
        W = self.generate_confidence_weights(X)

        # Reshape về batch format
        X_scaled = X_scaled.view(B, T, K, 2)  # (B,T,K,2)

        # Nếu input ban đầu không có batch, trả về không có batch
        if not has_batch:
            X_scaled = X_scaled.squeeze(0)  # (T,K,2)

        return X_scaled, W


# Unit test / example usage
if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2  # batch size
    T = 32
    K = 21
    d = 64
    d_b = 16

    parts = {
        "body": list(range(0, 7)),
        "lh": list(range(7, 14)),
        "rh": list(range(14, 21)),
    }

    pre = Preprocessor(
        index={"body": 0, "left": 1, "right": 2},
        parts=parts,
        conf_alpha=1.0,
    )

    # Test với batch size
    X_batch = torch.randn(B, T, K, 2) * 100.0

    X_bar_batch, W_batch = pre(X_batch)
    print("Batch input - X_bar.shape:", X_bar_batch.shape)
    print("Batch input - W.shape:", W_batch.shape)

    # Test không có batch size
    X_no_batch = torch.randn(T, K, 2) * 100.0

    X_bar_no_batch, W_no_batch = pre(X_no_batch)
    print("No batch - X_bar.shape:", X_bar_no_batch.shape)
    print("No batch - W.shape:", W_no_batch.shape)

    print("Preprocessor params:", count_parameters(pre))
    print("Total params (these modules):", count_parameters(pre))

    assert torch.isfinite(X_bar_batch).all(), "X_bar_batch contains NaN/Inf"
    assert torch.isfinite(X_bar_no_batch).all(), "X_bar_no_batch contains NaN/Inf"
    print("Sanity checks passed.")
