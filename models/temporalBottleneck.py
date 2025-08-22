# Re-run the Temporal Bottleneck implementation and tests (the previous cell was reset).
import torch
import torch.nn as nn
import math

torch.manual_seed(0)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.h = heads
        self.d_h = d // heads
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.o_proj = nn.Linear(d, d, bias=True)

    def forward(self, Q, K, V):
        B, Lq, d = Q.shape
        Bk, Lk, dk = K.shape
        assert d == self.d and dk == self.d and B == Bk

        q = self.q_proj(Q).view(B, Lq, self.h, self.d_h).transpose(1, 2)  # (B,h,Lq,d_h)
        k = self.k_proj(K).view(B, Lk, self.h, self.d_h).transpose(1, 2)  # (B,h,Lk,d_h)
        v = self.v_proj(V).view(B, Lk, self.h, self.d_h).transpose(1, 2)  # (B,h,Lk,d_h)

        scores = torch.einsum("bhlp,bhmp->bhlm", q, k) / math.sqrt(
            self.d_h
        )  # (B,h,Lq,Lk)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhlm,bhmp->bhlp", attn, v)  # (B,h,Lq,d_h)
        out = out.transpose(1, 2).contiguous().view(B, Lq, d)  # (B,Lq,d)
        out = self.o_proj(out)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TemporalBottleneckLayer(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, expand: bool):
        super().__init__()
        self.expand = expand
        # Z <- S
        self.lnZ1 = nn.LayerNorm(d)
        self.lnS1 = nn.LayerNorm(d)
        self.ca_Z_from_S = MultiHeadAttention(d, heads)
        self.ffnZ = FFN(d, d_ff)
        self.lnZ2 = nn.LayerNorm(d)
        # self-attn on Z
        self.sa_Z = MultiHeadAttention(d, heads)
        self.lnZ3 = nn.LayerNorm(d)
        # S branch (only if expand)
        if expand:
            self.lnS2 = nn.LayerNorm(d)
            self.lnZ4 = nn.LayerNorm(d)
            self.ca_S_from_Z = MultiHeadAttention(d, heads)
            self.ffnS = FFN(d, d_ff)
            self.lnS3 = nn.LayerNorm(d)

    def forward(self, Z, S):
        # Cross-attn: Z <- S
        Z1 = self.lnZ1(Z)
        S1 = self.lnS1(S)
        Z = Z + self.ca_Z_from_S(Z1, S1, S1)
        Z = Z + self.ffnZ(self.lnZ2(Z))
        # Self-attn on Z
        Z = Z + self.sa_Z(self.lnZ3(Z), self.lnZ3(Z), self.lnZ3(Z))
        if self.expand:
            S2 = self.lnS2(S)
            Z2 = self.lnZ4(Z)
            S = S + self.ca_S_from_Z(S2, Z2, Z2)
            S = S + self.ffnS(self.lnS3(S))
        return Z, S


class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d)

    def forward(self, S):
        B, T, d = S.shape
        pos = torch.arange(T, device=S.device).unsqueeze(0).expand(B, T)
        return S + self.pe(pos)


class TemporalBottleneck(nn.Module):
    def __init__(
        self,
        d: int = 64,
        m_slots: int = 8,
        heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        mode: str = "bottleneck",
        d_out: int = None,
        use_positional: bool = True,
        max_len: int = 512,
    ):
        super().__init__()
        assert mode in ("bottleneck", "expand")
        self.d = d
        self.M = m_slots
        self.mode = mode
        self.use_positional = use_positional

        self.Z0 = nn.Parameter(torch.randn(m_slots, d) * 0.02)

        if use_positional:
            self.pos = TemporalPositionalEmbedding(max_len=max_len, d=d)
        else:
            self.pos = None

        self.layers = nn.ModuleList(
            [
                TemporalBottleneckLayer(
                    d=d, heads=heads, d_ff=d_ff, expand=(mode == "expand")
                )
                for _ in range(num_layers)
            ]
        )

        if d_out is not None and d_out != d:
            self.proj_out_Z = nn.Linear(d, d_out, bias=True)
            if mode == "expand":
                self.proj_out_S = nn.Linear(d, d_out, bias=True)
            else:
                self.proj_out_S = None
        else:
            self.proj_out_Z = None
            self.proj_out_S = None

    def forward(self, S):
        B, T, d = S.shape
        assert d == self.d

        if self.pos is not None:
            S = self.pos(S)

        Z = self.Z0.unsqueeze(0).expand(B, self.M, d)
        S_orig = S.clone()  # Lưu S ban đầu

        for layer in self.layers:
            Z, S = layer(Z, S)

        if self.mode == "bottleneck":
            if self.proj_out_Z is not None:
                Z = self.proj_out_Z(Z)
            return Z, S_orig  # Trả về Z và S ban đầu
        else:
            if self.proj_out_S is not None:
                S = self.proj_out_S(S)
            return S, S  # Trả về S và S (để giữ interface)


# ---------------- Tests ----------------
if __name__ == "__main__":
    torch.manual_seed(1)
    B = 2
    T = 40
    d = 64
    M = 8
    heads = 4
    d_ff = 128
    L = 2

    S = torch.randn(B, T, d)

    tb_bottleneck = TemporalBottleneck(
        d=d,
        m_slots=M,
        heads=heads,
        d_ff=d_ff,
        num_layers=L,
        mode="bottleneck",
        d_out=64,
        use_positional=True,
        max_len=256,
    )
    Zb, Sb = tb_bottleneck(S)
    print("[Bottleneck] Zb.shape:", Zb.shape)
    print("[Bottleneck] Sb.shape:", Sb.shape)
    print("TemporalBottleneck (bottleneck) params:", count_parameters(tb_bottleneck))

    assert Zb.shape == (B, M, 64)
    assert Sb.shape == (B, T, d)
    assert torch.isfinite(Zb).all()
    assert torch.isfinite(Sb).all()

    tb_expand = TemporalBottleneck(
        d=d,
        m_slots=M,
        heads=heads,
        d_ff=d_ff,
        num_layers=L,
        mode="expand",
        d_out=64,
        use_positional=True,
        max_len=256,
    )
    Se, Se2 = tb_expand(S)
    print("[Expand] Se.shape:", Se.shape)
    print("[Expand] Se2.shape:", Se2.shape)
    print("TemporalBottleneck (expand) params:", count_parameters(tb_expand))

    assert Se.shape == (B, T, 64)
    assert Se2.shape == (B, T, 64)
    assert torch.isfinite(Se).all()
    assert torch.isfinite(Se2).all()

    print("All TemporalBottleneck tests passed.")
