import torch, torch.nn as nn, torch.nn.functional as F
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, h, batch_first=True); self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False); x = x + a
        return x + self.mlp(self.ln2(x))
class TinyGPT(nn.Module):
    def __init__(self, vocab=20, seq_len=10, d=128, h=4, n_layers=4):
        super().__init__(); self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)]); self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1))
    def forward(self, x, return_hidden=False):
        T = x.shape[1]; hcur = self.tok(x) + self.pos(torch.arange(T, device=x.device)); hs = [hcur]
        for b in self.blocks:
            hcur = b(hcur, self.mask[:T, :T]); hs.append(hcur)
        logits = self.head(self.ln(hcur))
        return (logits, hs) if return_hidden else logits
