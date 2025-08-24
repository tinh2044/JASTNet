import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RelativeBiasMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        orig_shape = D.shape
        x = D.reshape(-1, 1)
        y = self.net(x)
        return y.view(orig_shape)


class SpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        d: int,
        heads: int,
        d_ff: int,
        rel_bias_hidden: int = 32,
        conf_phi_tau: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d % heads == 0, "d must be divisible by heads"
        self.d = d
        self.h = heads
        self.d_h = d // heads
        self.eps = eps
        self.conf_phi_tau = conf_phi_tau

        self.qkv_proj = nn.Linear(d, 3 * d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn_fc1 = nn.Linear(d, d_ff)
        self.ffn_fc2 = nn.Linear(d_ff, d)
        self.act = nn.GELU()

        self.rel_bias = RelativeBiasMLP(hidden=rel_bias_hidden)

    def phi(self, c: torch.Tensor) -> torch.Tensor:
        t = self.conf_phi_tau
        return (c / (c + t)).clamp(0.0, 1.0)

    def forward(
        self, X_in: torch.Tensor, barX_t: torch.Tensor, W_t: torch.Tensor
    ) -> torch.Tensor:
        B, T, K, d = X_in.shape
        assert d == self.d

        X_norm = self.ln1(X_in)  # (B, T, K, d)
        qkv = self.qkv_proj(X_norm)  # (B, T, K, 3d)
        qkv = qkv.view(B, T, K, 3, self.h, self.d_h)  # (B, T, K, 3, H, d_h)
        # select q, k, v and permute to (B, T, H, K, d_h)
        q = qkv[:, :, :, 0].permute(0, 1, 3, 2, 4).contiguous()  # (B, T, H, K, d_h)
        k = qkv[:, :, :, 1].permute(0, 1, 3, 2, 4).contiguous()  # (B, T, H, K, d_h)
        v = qkv[:, :, :, 2].permute(0, 1, 3, 2, 4).contiguous()  # (B, T, H, K, d_h)

        scores = torch.einsum("bthkd,bthld->bthkl", q, k)  # (B, T, H, K, K)
        scores = scores / math.sqrt(self.d_h)

        g = self.phi(W_t)  # (B, T, K)
        gg = g.unsqueeze(3) * g.unsqueeze(2)  # (B, T, K, K)
        M_t = torch.log(gg + self.eps)  # (B, T, K, K)
        M_t = M_t.unsqueeze(2)  # (B, T, 1, K, K)
        scores = scores + M_t

        diff = barX_t.unsqueeze(3) - barX_t.unsqueeze(2)  # (B, T, K, K, 2)
        D = torch.norm(diff, dim=-1)  # (B, T, K, K)
        B_t = self.rel_bias(D)  # (B, T, K, K)
        B_t = B_t.unsqueeze(2)  # (B, T, 1, K, K)
        scores = scores + B_t

        attn = torch.softmax(scores, dim=-1)  # (B, T, H, K, K)

        out_heads = torch.einsum("bthkl,bthld->bthkd", attn, v)  # (B, T, H, K, d_h)
        out_heads = (
            out_heads.permute(0, 1, 3, 2, 4).contiguous().view(B, T, K, d)
        )  # (B, T, K, d)

        attn_out = self.out_proj(out_heads)  # (B, T, K, d)
        Y = X_in + attn_out  # residual

        Y_ln = self.ln2(Y)  # (B, T, K, d)
        ff = self.act(self.ffn_fc1(Y_ln))  # (B, T, K, d_ff)
        ff = self.ffn_fc2(ff)  # (B, T, K, d)
        Z = Y + ff  # (B, T, K, d)

        return Z


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        d: int = 64,
        heads: int = 4,
        L_s: int = 2,
        d_ff: int = 256,
        rel_bias_hidden: int = 32,
        conf_phi_tau: float = 1e-3,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SpatialAttentionLayer(
                    d=d,
                    heads=heads,
                    d_ff=d_ff,
                    rel_bias_hidden=rel_bias_hidden,
                    conf_phi_tau=conf_phi_tau,
                )
                for _ in range(L_s)
            ]
        )

    def forward(
        self, E0: torch.Tensor, barX: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        if E0.dim() == 3:
            E0 = E0.unsqueeze(0)  # (1,T,K,d)
            barX = barX.unsqueeze(0)  # (1,T,K,2)
            W = W.unsqueeze(0)  # (1,T,K)
            has_batch = False
        else:
            has_batch = True

        x = E0
        for layer in self.layers:
            x = layer(x, barX, W)

        if not has_batch:
            x = x.squeeze(0)  # (T,K,d)

        return x


if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2
    T = 16
    K = 21
    d = 64
    heads = 4
    L_s = 2
    d_ff = 128

    E0_batch = torch.randn(B, T, K, d)
    barX_batch = torch.randn(B, T, K, 2)
    W_batch = torch.rand(B, T, K)

    encoder = SpatialEncoder(d=d, heads=heads, L_s=L_s, d_ff=d_ff)
    E_S_batch = encoder(E0_batch, barX_batch, W_batch)

    print("Batch input - E0.shape:", E0_batch.shape)
    print("Batch input - barX.shape:", barX_batch.shape)
    print("Batch input - W.shape:", W_batch.shape)
    print("Batch output - E_S.shape:", E_S_batch.shape)
    print("Expected shape: ({}, {}, {}, {})".format(B, T, K, d))

    E0_no_batch = torch.randn(T, K, d)
    barX_no_batch = torch.randn(T, K, 2)
    W_no_batch = torch.rand(T, K)

    E_S_no_batch = encoder(E0_no_batch, barX_no_batch, W_no_batch)

    print("\nNo batch - E0.shape:", E0_no_batch.shape)
    print("No batch - barX.shape:", barX_no_batch.shape)
    print("No batch - W.shape:", W_no_batch.shape)
    print("No batch output - E_S.shape:", E_S_no_batch.shape)
    print("Expected shape: ({}, {}, {})".format(T, K, d))

    print("\nSpatialEncoder params:", count_parameters(encoder))
    assert torch.isfinite(E_S_batch).all(), "E_S_batch contains NaN/Inf"
    assert torch.isfinite(E_S_no_batch).all(), "E_S_no_batch contains NaN/Inf"
    print("Sanity checks passed for SpatialEncoder.")
