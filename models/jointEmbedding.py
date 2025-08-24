from typing import Dict, List, Optional
import torch
import torch.nn as nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class JointEmbedding(nn.Module):
    def __init__(
        self,
        K: int,
        d: int = 64,
        d_bottleneck: int = 16,
        parts_order: List[str] = ["body", "lh", "rh"],
        part_to_indices: Optional[Dict[str, List[int]]] = None,
        conf_phi_tau: float = 1e-3,
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.d_bottleneck = d_bottleneck
        self.parts_order = parts_order
        self.part_to_indices = part_to_indices or {}

        self.W_x = nn.Parameter(torch.randn(d, 1) * (1.0 / (d**0.5)))
        self.W_y = nn.Parameter(torch.randn(d, 1) * (1.0 / (d**0.5)))
        self.b_e = nn.Parameter(torch.zeros(d))

        self.joint_emb = nn.Parameter(torch.randn(K, d) * 0.02)
        self.part_emb = nn.Parameter(torch.randn(len(parts_order), d) * 0.02)

        self.fuse_fc1 = nn.Linear(d, d_bottleneck)
        self.fuse_fc2 = nn.Linear(d_bottleneck, d)
        self.act = nn.GELU()

        self.ln = nn.LayerNorm(d)
        self.W_out = nn.Linear(d, d)

        self.conf_phi_tau = conf_phi_tau

    def phi(self, c: torch.Tensor) -> torch.Tensor:
        t = self.conf_phi_tau
        return (c / (c + t)).clamp(0.0, 1.0)

    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3:
            X = X.unsqueeze(0)  # (1,T,K,2)
            W = W.unsqueeze(0)  # (1,T,K)
            has_batch = False
        else:
            has_batch = True

        assert X.dim() == 4 and X.size(3) == 2, (
            f"X must be of shape (B, T, K, 2) or (T, K, 2), got {X.shape}"
        )
        B, T, K, _ = X.shape
        assert K == self.K, f"JointEmbedding K mismatch: got {K} vs init {self.K}"

        x = X[:, :, :, 0].unsqueeze(-1)  # (B,T,K,1)
        y = X[:, :, :, 1].unsqueeze(-1)  # (B,T,K,1)

        p = torch.matmul(x, self.W_x.T) + torch.matmul(y, self.W_y.T)  # (T,K,d)
        p = p + self.b_e.view(1, 1, -1)

        joint_emb_exp = self.joint_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        q = p + joint_emb_exp

        if len(self.part_to_indices) > 0:
            part_idx_for_joint = torch.full((K,), -1, dtype=torch.long, device=X.device)
            for pid, pname in enumerate(self.parts_order):
                idxs = self.part_to_indices.get(pname, [])
                if len(idxs) > 0:
                    part_idx_for_joint[idxs] = pid
            part_emb_per_joint = torch.zeros(K, self.d, device=X.device)
            valid_mask = part_idx_for_joint >= 0
            if valid_mask.any():
                part_emb_per_joint[valid_mask] = self.part_emb[
                    part_idx_for_joint[valid_mask]
                ]
            part_emb_exp = (
                part_emb_per_joint.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            )
            r = q + part_emb_exp
        else:
            r = q

        g = self.phi(W).unsqueeze(-1)  # (B,T,K,1)
        s = g * r  # (B,T,K,d)

        s_flat = s.view(B * T * K, self.d)
        hidden = self.act(self.fuse_fc1(s_flat))  # (T*K, d_bottleneck)
        out = self.fuse_fc2(hidden)  # (T*K, d)
        out = out.view(B, T, K, self.d)
        u = s + out  # residual connection

        u_ln = self.ln(u)
        E0 = self.W_out(u_ln)  # (B,T,K,d)

        if not has_batch:
            E0 = E0.squeeze(0)  # (T,K,d)

        return E0


if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2
    T = 32
    K = 21
    d = 64
    d_b = 16

    parts = {
        "body": list(range(0, 7)),
        "lh": list(range(7, 14)),
        "rh": list(range(14, 21)),
    }
    je = JointEmbedding(
        K=K,
        d=d,
        d_bottleneck=d_b,
        parts_order=["body", "lh", "rh"],
        part_to_indices=parts,
        conf_phi_tau=1e-3,
    )

    X_batch = torch.randn(B, T, K, 2) * 100.0
    C_batch = torch.rand(B, T, K)

    E0_batch = je(X_batch, C_batch)
    print("Batch input - E0.shape:", E0_batch.shape)
    print("Expected shape: ({}, {}, {}, {})".format(B, T, K, d))

    X_no_batch = torch.randn(T, K, 2) * 100.0
    C_no_batch = torch.rand(T, K)

    E0_no_batch = je(X_no_batch, C_no_batch)
    print("No batch - E0.shape:", E0_no_batch.shape)
    print("Expected shape: ({}, {}, {})".format(T, K, d))

    print("JointEmbedding params:", count_parameters(je))
    assert torch.isfinite(E0_batch).all(), "E0_batch contains NaN/Inf"
    assert torch.isfinite(E0_no_batch).all(), "E0_no_batch contains NaN/Inf"
    print("Sanity checks passed.")
