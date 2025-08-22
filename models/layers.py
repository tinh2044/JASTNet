import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

torch.manual_seed(0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class MHA(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0, "d must be divisible by heads"
        self.d = d
        self.h = heads
        self.d_h = d // heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, Q, K, V, mask: Optional[torch.Tensor] = None):
        # Q: (B, Lq, d), K,V: (B, Lk, d); mask optional (B, Lq, Lk) additive (logits)
        B, Lq, d = Q.shape
        Bk, Lk, dk = K.shape
        assert d == self.d and dk == d and B == Bk
        q = self.q_proj(Q).view(B, Lq, self.h, self.d_h).transpose(1, 2)  # (B,h,Lq,d_h)
        k = self.k_proj(K).view(B, Lk, self.h, self.d_h).transpose(1, 2)  # (B,h,Lk,d_h)
        v = self.v_proj(V).view(B, Lk, self.h, self.d_h).transpose(1, 2)  # (B,h,Lk,d_h)
        scores = torch.einsum("bhlp,bhmp->bhlm", q, k) / math.sqrt(
            self.d_h
        )  # (B,h,Lq,Lk)
        if mask is not None:
            # mask expected shape (B, Lq, Lk) or broadcastable; expand to heads
            scores = scores + mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhlm,bhmp->bhlp", attn, v)  # (B,h,Lq,d_h)
        out = out.transpose(1, 2).contiguous().view(B, Lq, d)  # (B,Lq,d)
        return self.o_proj(out)


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MotionResidualBranch(nn.Module):
    """
    Input:
      barX: (B, T, K, 2)  -- preprocessed coordinates
      W:    (B, T, K)     -- confidences
      parts: dict mapping part_name -> list of joint indices (e.g., body, lh, rh)
    Output:
      R: (B, T, d) -- motion residual per frame (same d as S)
    """

    def __init__(self, d: int, parts: dict, kernel_size: int = 3, **cfg):
        super().__init__()
        assert kernel_size % 2 == 1
        self.d = d
        self.parts = parts
        self.K = sum(len(v) for v in parts.values())
        # linear projectors from 2->d for velocity and acceleration
        self.phi_v = nn.Linear(2, d)
        self.phi_a = nn.Linear(2, d)
        # scalar gating coefficients (trainable)
        self.gamma_v = nn.Parameter(torch.tensor(1.0))
        self.gamma_a = nn.Parameter(torch.tensor(0.5))
        # part aggregator projection
        self.WM = nn.Linear(len(parts) * d, d)
        # depthwise conv (implemented as group conv) for temporal smoothing
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        # Depthwise conv1d: in_channels=d, out_channels=d, groups=d
        self.dwconv = nn.Conv1d(
            in_channels=d,
            out_channels=d,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=d,
        )
        # pointwise projection
        self.pw = nn.Linear(d, d)
        # layernorm
        self.ln = nn.LayerNorm(d)

    def forward(self, barX: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # barX: (B,T,K,2), W: (B,T,K)
        B, T, K, _ = barX.shape
        assert K == self.K
        # compute first and second differences along time
        # pad at t=0 with zeros for delta1 and delta2
        X_t = barX
        X_tm1 = torch.cat(
            [
                torch.zeros(B, 1, K, 2, device=barX.device, dtype=barX.dtype),
                barX[:, :-1],
            ],
            dim=1,
        )
        V = X_t - X_tm1  # (B,T,K,2)
        V_tm1 = torch.cat(
            [torch.zeros(B, 1, K, 2, device=barX.device, dtype=barX.dtype), V[:, :-1]],
            dim=1,
        )
        A = V - V_tm1  # (B,T,K,2)

        # project per joint
        v_proj = self.phi_v(V)  # (B,T,K,d)
        a_proj = self.phi_a(A)  # (B,T,K,d)
        # combine with learnable weights
        Hm = self.gamma_v * v_proj + self.gamma_a * a_proj  # (B,T,K,d)

        # confidence weighting normalized per frame (over all joints)
        W_sum = W.sum(dim=2, keepdim=True) + 1e-6  # (B,T,1)
        w_norm = W / W_sum  # (B,T,K)
        w_norm = w_norm.unsqueeze(-1)  # (B,T,K,1)
        Hm_weighted = Hm * w_norm  # (B,T,K,d)

        # per-part pooling (sum over indices in each part)
        part_vecs = []
        for p_name, idxs in self.parts.items():
            if len(idxs) == 0:
                part_vecs.append(
                    torch.zeros(B, T, self.d, device=barX.device, dtype=barX.dtype)
                )
                continue
            sel = Hm_weighted[:, :, idxs, :]  # (B,T, Pp, d)
            pooled = sel.sum(dim=2)  # (B,T,d)
            part_vecs.append(pooled)
        # concat parts -> (B,T, P*d)
        concat = torch.cat(part_vecs, dim=2)  # (B,T, P*d)
        # linear projection to d
        U = self.WM(concat)  # (B,T,d)
        # temporal depthwise conv: expect (B,d,T) input for conv1d
        U_t = U.permute(0, 2, 1)  # (B,d,T)
        U_conv = self.dwconv(U_t)  # (B,d,T)
        U_conv = U_conv.permute(0, 2, 1)  # (B,T,d)
        U_pw = self.pw(U_conv)  # (B,T,d)
        U_ln = self.ln(U_pw)
        # output residual R (ready to add to S)
        R = U_ln  # (B,T,d)
        return R


class CrossFusionMemoryMotion(nn.Module):
    """
    Cross-fusion between memory slots Z (B,M,d) and motion stream R (B,T,d)
    Produces updated Z_star (B,M,d) and fused sequence F (B,T,d).
    Config:
      - heads: attention heads
      - ff_dims: dict with 'Z' and 'R' ff dims
      - layers: number of fusion layers (repeat)
    """

    def __init__(
        self, d: int, heads: int, ff_dim_z: int, ff_dim_r: int, layers: int = 1
    ):
        super().__init__()
        self.d = d
        self.layers = layers
        self.fusion_layers = nn.ModuleList()
        for _ in range(layers):
            layer = nn.ModuleDict(
                {
                    "lnZ": nn.LayerNorm(d),
                    "lnR": nn.LayerNorm(d),
                    "ca_Z_from_R": MHA(d, heads),
                    "ffnZ": FFN(d, ff_dim_z),
                    "lnZ2": nn.LayerNorm(d),
                    "lnR2": nn.LayerNorm(d),
                    "ca_R_from_Z": MHA(d, heads),
                    "ffnR": FFN(d, ff_dim_r),
                }
            )
            self.fusion_layers.append(layer)

    def forward(
        self, Z: torch.Tensor, R: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Z: (B, M, d), R: (B, T, d)
        B, M, d = Z.shape
        _, T, _ = R.shape
        assert d == self.d
        for layer in self.fusion_layers:
            # Z <- R
            Z_ln = layer["lnZ"](Z)
            R_ln = layer["lnR"](R)
            deltaZ = layer["ca_Z_from_R"](Z_ln, R_ln, R_ln)  # (B,M,d)
            Z = Z + deltaZ
            Z = Z + layer["ffnZ"](layer["lnZ2"](Z))

            # R <- Z
            R_ln2 = layer["lnR2"](R)
            Z_ln2 = layer["lnZ2"](Z)
            deltaR = layer["ca_R_from_Z"](R_ln2, Z_ln2, Z_ln2)  # (B,T,d)
            R = R + deltaR
            R = R + layer["ffnR"](layer["lnR2"](R))

        return Z, R


class ClassificationHead(nn.Module):
    """
    Pools sequence F (B,T,d) using a learned query, optionally concatenates pooled memory Z,
    and produces logits (B,C).
    """

    def __init__(self, d: int, num_classes: int, use_memory: bool = True):
        super().__init__()
        self.d = d
        self.C = num_classes
        self.use_memory = use_memory
        self.query = nn.Parameter(torch.randn(d) * 0.02)  # learned query
        if use_memory:
            self.mem_proj = nn.Linear(2 * d, d)
        self.cls = nn.Linear(d, num_classes)

    def forward(
        self, F: torch.Tensor, Z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # F: (B,T,d), Z: (B,M,d) or None
        B, T, d = F.shape
        assert d == self.d
        # compute attention weights with query
        F_ln = F  # can use LN if desired outside
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,d)
        q = q.expand(B, 1, d)  # (B,1,d)
        # dot product pooling: compute scores for each t
        scores = torch.matmul(F_ln, q.transpose(1, 2)).squeeze(-1)  # (B,T)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B,T,1)
        hF = (alpha * F).sum(dim=1)  # (B,d)
        if self.use_memory and Z is not None:
            hZ = Z.mean(dim=1)  # (B,d)
            h = torch.cat([hF, hZ], dim=-1)  # (B,2d)
            h = self.mem_proj(h)  # (B,d)
        else:
            h = hF
        logits = self.cls(h)  # (B,C)
        return logits


if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2
    T = 32
    K = 21
    d = 64
    M = 8

    barX = torch.randn(B, T, K, 2)
    W = torch.rand(B, T, K)
    S = torch.randn(B, T, d)
    Z = torch.randn(B, M, d)

    parts = {
        "body": list(range(0, 7)),
        "lh": list(range(7, 14)),
        "rh": list(range(14, 21)),
    }

    motion = MotionResidualBranch(d=d, parts=parts, kernel_size=3)
    R = motion(barX, W)
    print("Motion R.shape:", R.shape, "Params:", count_parameters(motion))
    assert R.shape == (B, T, d)

    fusion = CrossFusionMemoryMotion(d=d, heads=4, ff_dim_z=128, ff_dim_r=128, layers=1)
    Z2, R2 = fusion(Z, R)
    print(
        "Fusion Z2.shape:",
        Z2.shape,
        "R2.shape:",
        R2.shape,
        "Params:",
        count_parameters(fusion),
    )
    assert Z2.shape == (B, M, d) and R2.shape == (B, T, d)

    head = ClassificationHead(d=d, num_classes=200, use_memory=True)
    logits = head(R2, Z2)
    print("Logits.shape:", logits.shape, "Params:", count_parameters(head))
    assert logits.shape == (B, 200)
    assert torch.isfinite(logits).all()
