# Re-run FrameSummary code and test to produce outputs.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

torch.manual_seed(0)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FrameSummary(nn.Module):
    def __init__(
        self,
        d: int,
        parts_order: List[str],
        part_to_indices: Dict[str, List[int]],
        eps: float = 1e-6,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.d = d
        self.parts_order = parts_order
        self.part_to_indices = part_to_indices
        self.eps = eps
        self.P = len(parts_order)
        self.use_layernorm = use_layernorm

        self.part_linears = nn.ModuleDict(
            {p: nn.Linear(d, d, bias=True) for p in parts_order}
        )

        self.v = nn.ParameterDict(
            {p: nn.Parameter(torch.randn(d) * 0.02) for p in parts_order}
        )
        self.c = nn.ParameterDict(
            {p: nn.Parameter(torch.zeros(1)) for p in parts_order}
        )

        self.proj = nn.Linear(self.P * d, d, bias=True)

        if use_layernorm:
            self.part_ln = nn.ModuleDict({p: nn.LayerNorm(d) for p in parts_order})
            self.out_ln = nn.LayerNorm(d)
        else:
            self.part_ln = None
            self.out_ln = None

    def forward(self, E_S: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # E_S: (B,T,K,d) hoặc (T,K,d), W: (B,T,K) hoặc (T,K)
        if E_S.dim() == 3:
            # Không có batch size, thêm batch dimension
            E_S = E_S.unsqueeze(0)  # (1,T,K,d)
            W = W.unsqueeze(0)  # (1,T,K)
            has_batch = False
        else:
            has_batch = True

        assert E_S.dim() == 4 and W.dim() == 3
        B, T, K, d = E_S.shape
        assert d == self.d, "Embedding dim mismatch"

        part_vectors = []
        for p in self.parts_order:
            idxs = self.part_to_indices.get(p, [])
            if len(idxs) == 0:
                u_tp = torch.zeros(B, T, d, device=E_S.device, dtype=E_S.dtype)
                part_vectors.append(u_tp)
                continue

            tokens = E_S[:, :, idxs, :]  # (B,T, Pp, d)
            weights = W[:, :, idxs]  # (B,T, Pp)

            denom = weights.sum(dim=2, keepdim=True)  # (B,T,1)
            denom = denom + self.eps
            w_norm = weights / denom  # (B,T, Pp)

            w_exp = w_norm.unsqueeze(-1)  # (B,T, Pp, 1)
            u_tp = (w_exp * tokens).sum(dim=2)  # (B,T, d)

            u_hat = self.part_linears[p](u_tp)  # (B,T, d)

            if self.use_layernorm:
                u_hat_ln = self.part_ln[p](u_hat)  # (B,T, d)
            else:
                u_hat_ln = u_hat

            v_p = self.v[p]  # (d,)
            c_p = self.c[p]  # (1,)
            dot = (u_hat_ln * v_p.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B,T)
            alpha = torch.sigmoid(dot + c_p)  # (B,T)
            alpha = alpha.unsqueeze(-1)  # (B,T,1)

            u_tilde = alpha * u_hat  # (B,T, d)
            part_vectors.append(u_tilde)

        Z = torch.cat(part_vectors, dim=-1)  # (B,T, P*d)
        S = self.proj(Z)  # (B,T, d)

        if self.out_ln is not None:
            S = self.out_ln(S)

        # Nếu input ban đầu không có batch, trả về không có batch
        if not has_batch:
            S = S.squeeze(0)  # (T,d)

        return S


# Test
if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2  # batch size
    T = 32
    K = 21
    d = 64

    parts = {
        "body": list(range(0, 7)),
        "lh": list(range(7, 14)),
        "rh": list(range(14, 21)),
    }
    parts_order = ["body", "lh", "rh"]

    # Test với batch size
    E_S_batch = torch.randn(B, T, K, d)
    W_batch = torch.rand(B, T, K)

    frame_summary = FrameSummary(
        d=d, parts_order=parts_order, part_to_indices=parts, use_layernorm=True
    )
    S_batch = frame_summary(E_S_batch, W_batch)

    print("Batch input - E_S.shape:", E_S_batch.shape)
    print("Batch input - W.shape:", W_batch.shape)
    print("Batch output - S.shape:", S_batch.shape)
    print("Expected shape: ({}, {}, {})".format(B, T, d))

    # Test không có batch size
    E_S_no_batch = torch.randn(T, K, d)
    W_no_batch = torch.rand(T, K)

    S_no_batch = frame_summary(E_S_no_batch, W_no_batch)

    print("\nNo batch - E_S.shape:", E_S_no_batch.shape)
    print("No batch - W.shape:", W_no_batch.shape)
    print("No batch output - S.shape:", S_no_batch.shape)
    print("Expected shape: ({}, {})".format(T, d))

    print("\nFrameSummary params:", count_parameters(frame_summary))
    assert torch.isfinite(S_batch).all(), "S_batch contains NaN/Inf"
    assert torch.isfinite(S_no_batch).all(), "S_no_batch contains NaN/Inf"
    print("FrameSummary sanity checks passed.")
