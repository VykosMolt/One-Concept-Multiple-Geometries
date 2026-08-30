"""Hidden-state extraction for concept families under prompt templates.

For each template (a string with '{x}'), each concept string x is inserted; we tokenize with offset
mapping to locate the concept span. Positions extracted per layer:
  'last'   : last token of the concept span
  'mean'   : mean over concept-span tokens
  'anchor' : first token after the concept span (identical string across concepts if the template
             has post-concept text) — None if the concept is prompt-final
  'final'  : final token of the prompt (Karkada A.3 convention)
Returns dict pos -> array (n_layers+1, n_concepts, d) in float32. Layer 0 = embedding output.
"""
from __future__ import annotations
import torch, numpy as np


class Extractor:
    def __init__(self, model_path: str, dtype=torch.float32, device="cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.n_layers = self.model.config.num_hidden_layers

    def tokenize_with_span(self, template: str, x: str, add_bos: bool = True):
        pre, post = template.split("{x}")
        text = pre + x + post
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=add_bos)
        ids = enc["input_ids"]; offs = enc["offset_mapping"]
        cs, ce = len(pre), len(pre) + len(x)
        span = [i for i, (a, b) in enumerate(offs) if b > cs and a < ce and b > a]
        after = [i for i, (a, b) in enumerate(offs) if a >= ce and b > a]
        return text, ids, span, (after[0] if after else None)

    def describe_tokens(self, template: str, x: str):
        text, ids, span, anchor = self.tokenize_with_span(template, x)
        toks = self.tok.convert_ids_to_tokens(ids)
        return {"text": text, "tokens": toks, "span_tokens": [toks[i] for i in span], "n_span": len(span),
                "anchor_token": toks[anchor] if anchor is not None else None}

    @torch.no_grad()
    def hidden_states(self, text_ids: list[int]):
        ids = torch.tensor([text_ids], device=self.device)
        out = self.model(input_ids=ids, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)[:, 0]  # (L+1, T, d)
        return hs.float().cpu().numpy()

    def extract(self, template: str, concepts: list[str], positions=("last", "mean", "anchor", "final")):
        res = {p: [] for p in positions}
        meta = []
        for x in concepts:
            text, ids, span, anchor = self.tokenize_with_span(template, x)
            hs = self.hidden_states(ids)
            meta.append({"x": x, "n_span": len(span), "anchor": anchor is not None})
            for p in positions:
                if p == "last": res[p].append(hs[:, span[-1]])
                elif p == "mean": res[p].append(hs[:, span].mean(1))
                elif p == "anchor": res[p].append(hs[:, anchor] if anchor is not None else np.full_like(hs[:, 0], np.nan))
                elif p == "final": res[p].append(hs[:, -1])
        return {p: np.stack(v, 1) for p, v in res.items()}, meta

    def input_embeddings(self, concepts: list[str], prefix=" "):
        """Static input-embedding geometry: mean of token embeddings of ' x' (no context)."""
        E = self.model.get_input_embeddings().weight.detach().float().cpu().numpy()
        out, meta = [], []
        for x in concepts:
            ids = self.tok(prefix + x, add_special_tokens=False)["input_ids"]
            out.append(E[ids].mean(0)); meta.append({"x": x, "ids": ids, "tokens": self.tok.convert_ids_to_tokens(ids)})
        return np.stack(out), meta
