---
layout: post
title:  "attention residuals"
date:   2026-03-18 18:00:00 +0530
categories: tech
tokens: "~5k"
math: true
description: "Moonshot AI figured out that the residual connection - the most boring line in every transformer - is quietly sabotaging deep networks. The fix is 30 lines of PyTorch."
---

[Paper](https://arxiv.org/abs/2603.15031) | [GitHub](https://github.com/MoonshotAI/Attention-Residuals) - Moonshot AI (Kimi team)

*Prerequisites:* residual connections, [transformers](/tech/2025/11/27/ai.html), prenorm

---

## the one line nobody questioned

```python
x = x + Layer(x)
```

That's a residual connection. One line. Solved vanishing gradients in 2015, made 100+ layer networks trainable, and became *the* building block of deep learning. Every single modern transformer has this exact line:

```python
x = x + self.attn(self.norm1(x))  # this is in every LLM ever
```

It just works. So nobody ever touched it again.

But think about it - the weight on every layer's contribution is exactly **1.0**. Layer 1 gets the same importance as layer 64. The initial embedding gets the same weight as the final attention output. That's kinda dumb if you think about it.

Moonshot thought about it.

---

## three problems with `x = x + Layer(x)`

### 1. magnitude growth

64-layer transformer. After all layers, your hidden state is:

```
h_64 = x_0 + F_1(x_0) + F_2(h_1) + ... + F_64(h_63)
```

65 terms. All added with weight 1.0. What happens to the norm?

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
d = 512
n_layers = 64

x = torch.randn(1, 10, d)
layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])

norms = [x.norm().item()]
for layer in layers:
    with torch.no_grad():
        x = x + layer(x)
    norms.append(x.norm().item())

for i in [0, 8, 16, 32, 64]:
    print(f"After layer {i:2d}: norm = {norms[i]:.1f}")
```

Output: `72 -> 227 -> 725 -> 7241 -> 730954`

<video autoplay loop muted playsinline style="max-width:100%" preload="none">
  <source src="/assets/images/attn_res/norm_growth.mp4" type="video/mp4">
</video>

10,000x growth. Not because any layer is doing something insane - you're just dumping 64 outputs into a running sum and never normalizing.

### 2. information dilution

> imagine a meeting with 64 people. everyone speaks for exactly one minute. at the end you write a summary but you're forced to weight everyone equally.

That's standard residuals. After 64 layers, each layer contributes ~1.5% of the final hidden state. Layer 47 found a critical pattern between two tokens? Too bad, same weight as layer 3 which basically just copied the embedding.

<video autoplay loop muted playsinline style="max-width:100%" loading="lazy" preload="none">
  <source src="/assets/images/attn_res/dilution.mp4" type="video/mp4">
</video>

The deeper your network, the less any individual layer matters. 128-layer model doesn't give each layer twice the influence - it gives each layer *half*.

<style>
.dilution-widget{background:var(--bg-code,#f1f3f6);border-radius:8px;padding:16px 20px;margin:1rem 0;font-family:var(--font-mono,'JetBrains Mono',monospace)}
.dark-mode .dilution-widget{background:#1e1e2e}
.dilution-widget label{font-size:14px;display:block;margin-bottom:6px}
.dilution-widget input[type=range]{width:100%;margin:8px 0}
.dilution-widget .result{font-size:18px;font-weight:600;margin-top:8px}
.dilution-widget .result span{color:#ff6b6b}
</style>
<div class="dilution-widget">
  <label>Number of layers: <strong id="layer-count">64</strong></label>
  <input type="range" min="4" max="256" value="64" id="layer-slider" oninput="updateDilution()">
  <div class="result">Each layer contributes: <span id="layer-pct">1.5%</span></div>
</div>
<script>
function updateDilution(){var n=parseInt(document.getElementById('layer-slider').value);document.getElementById('layer-count').textContent=n;document.getElementById('layer-pct').textContent=(100/(n+1)).toFixed(1)+'%'}
</script>

### 3. prenorm makes it worse

You'd think prenorm fixes this - it normalizes input before each sub-layer:

```python
x = x + self.attn(self.norm1(x))  # norm bounds the input
```

The attention output is bounded. But you're adding that bounded output to `x`, which is the ever-growing running sum. Each new layer's signal is a smaller and smaller fraction of the total.

<video autoplay loop muted playsinline style="max-width:100%" loading="lazy" preload="none">
  <source src="/assets/images/attn_res/prenorm.mp4" type="video/mp4">
</video>

Last few layers of a 64-layer prenorm transformer are shouting into a hurricane. Bounded output, unbounded accumulator.

PreNorm solved gradient flow. Did not solve dilution. Made it worse.

---

## the fix: attention over depth

Instead of `weight = 1.0` for everything, let the network **learn** the weights.

How? Same way we learn everything else - attention. But instead of attending over sequence positions (the usual kind), attend over **depth**. Each block boundary looks at all previous checkpoint outputs and decides which ones matter.

<video autoplay loop muted playsinline style="max-width:100%" loading="lazy" preload="none">
  <source src="/assets/images/attn_res/weights_comparison.mp4" type="video/mp4">
</video>

Standard residual: every layer gets 12.5%. Attention residual: layer 2 gets 35%, layer 6 gets 25%, the rest share what's left. The network figures out what matters.

---

## the math (it's short)

Standard residual unfolds to equal-weight sum:

$$h_l = x_0 + F_1 + F_2 + \ldots + F_l$$

Attention residual replaces this with:

$$h_l = \sum_{i=0}^{l} \alpha_{i \to l} \cdot v_i$$

Where:
- $v_i = \text{RMSNorm}(h_i)$ - normalized output from layer $i$
- $\alpha_{i \to l} = \text{softmax}\left(\frac{w_l \cdot v_i}{\sqrt{d}}\right)$ - learned attention weight
- $w_l$ is a **learned parameter vector** per layer (not input-dependent)

RMSNorm here is subtle but important: it ensures scores depend on the *direction* of each layer's output, not its magnitude. Without it, deeper layers (with bigger norms) would dominate the softmax.

<video autoplay loop muted playsinline style="max-width:100%" loading="lazy" preload="none">
  <source src="/assets/images/attn_res/depth_attn.mp4" type="video/mp4">
</video>

$w_l$ is **zero-initialized**. When $w = 0$, all dot products are 0, softmax gives uniform weights. Start from democracy, learn to specialize. Same trick as LoRA zero-init.

---

## full attnres vs block attnres

**Full version**: every layer attends over all previous outputs. $O(L^2 d)$ compute, $O(Ld)$ memory. Clean but expensive.

**Block version** (what you actually use): partition 64 layers into blocks of 8. Standard residuals within blocks, depth attention at boundaries only.

- Layers 1-8: standard `x = x + Layer(x)`, output $h_8$
- **Boundary**: depth attention over $\{h_0, h_8\}$ -> $h_8'$
- Layers 9-16: standard residuals from $h_8'$, output $h_{16}$
- **Boundary**: depth attention over $\{h_0, h_8', h_{16}\}$ -> $h_{16}'$
- ...repeat

<video autoplay loop muted playsinline style="max-width:100%" loading="lazy" preload="none">
  <source src="/assets/images/attn_res/block_architecture.mp4" type="video/mp4">
</video>

Magnitude only grows within each 8-layer block, then gets reset by the softmax normalization at the boundary.

---

## the code

This is the whole thing. `DepthAttention` is ~15 lines, the rest is just wiring it into the existing transformer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DepthAttention(nn.Module):
    """Learned attention over layer depth at block boundaries."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))  # zero-init!
        self.norm = RMSNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, checkpoints):
        # checkpoints: list of (B, T, D) tensors
        V = torch.stack([self.norm(h) for h in checkpoints])  # (N, B, T, D)
        scores = torch.einsum('d, n b t d -> n b t', self.query, V) * self.scale
        weights = F.softmax(scores, dim=0)  # softmax over depth
        return (weights.unsqueeze(-1) * V).sum(dim=0)  # (B, T, D)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_head, mlp_hidden_dim, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.SiLU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerWithBlockAttnRes(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_head,
                 block_size=8, max_seq_len=1024):
        super().__init__()
        assert n_layers % block_size == 0
        self.block_size = block_size
        self.token_embeddings = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_head, 4 * dim, max_seq_len)
            for _ in range(n_layers)
        ])

        # one DepthAttention per block boundary
        n_blocks = n_layers // block_size
        self.depth_attns = nn.ModuleList([
            DepthAttention(dim) for _ in range(n_blocks)
        ])

        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.token_embeddings(idx)
        checkpoints = [x]  # h_0

        for i, layer in enumerate(self.layers):
            x = layer(x)  # standard residual inside TransformerBlock
            if (i + 1) % self.block_size == 0:
                block_idx = (i + 1) // self.block_size - 1
                x = self.depth_attns[block_idx](checkpoints + [x])
                checkpoints.append(x)

        x = self.norm_final(x)
        return self.lm_head(x)
```

```python
# test it
model = TransformerWithBlockAttnRes(vocab_size=1000, dim=256, n_layers=32, n_head=4, block_size=8)
idx = torch.randint(0, 1000, (2, 64))
out = model(idx)
print(f"Input:  {idx.shape}")                                # (2, 64)
print(f"Output: {out.shape}")                                # (2, 64, 1000)
print(f"Depth attention modules: {len(model.depth_attns)}")  # 4
```

The diff from standard Transformer++ is tiny. `TransformerBlock` is identical. You just add `DepthAttention` modules and change the forward loop to track checkpoints. ~30 lines of new code.

**Cost**: 4 block boundaries x 5 checkpoints max = 20 dot products per forward pass. Compare that to the billions of FLOPs in actual attention layers. ~2% overhead, <0.2% parameter increase.

---

## zero-init: why it doesn't break training

```python
import torch
import torch.nn.functional as F

w = torch.zeros(512)         # zero-initialized query
keys = torch.randn(8, 512)  # 8 checkpoint states

scores = (w @ keys.T) / (512 ** 0.5)
weights = F.softmax(scores, dim=0)

print("Weights:", [f"{v:.4f}" for v in weights.tolist()])
# [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]
```

$w = 0$ -> all dot products are 0 -> softmax gives uniform 1/8 weights. You start from averaging everything equally (which is reasonable), then the model learns to specialize. Same principle as LoRA and ResNet v2 zero-init: don't break what works, learn to improve it.

---

## results

> 7B to 70B params, 15T tokens

| Benchmark | Baseline | + AttnRes | Delta |
|-----------|----------|-----------|-------|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MBPP | 72.0 | 73.9 | +1.9 |

**1.25x compute savings** to match baseline quality. The +7.5 on GPQA-Diamond is wild for what's essentially 30 lines of code.

Caveat: these are Moonshot's internal runs at massive scale. [Code is open source](https://github.com/MoonshotAI/Attention-Residuals) so anyone can try it, but reproducing exact numbers requires their data/compute.

---

## transformer++ upgrade list

The running list of things that make modern LLMs better than the 2017 transformer:

- LayerNorm $\to$ RMSNorm
- GELU $\to$ SwiGLU
- Absolute PE $\to$ RoPE
- Post-Norm $\to$ Pre-Norm
- **Standard residual $\to$ Attention residual** (new)

The line `x = x + self.attn(self.norm1(x))` stays the same. The change is one level up - how block outputs get combined.

---

## closing

One line of code. Worked for 10 years. Nobody asked "why weight 1.0?". Everyone was busy optimizing layers - better attention, better norms, better activations, better positional encodings. The *connection* between layers? Just add with weight 1.0. Ship it.

Moonshot asked the question. The answer was attention - the same tool we use for everything else - but over depth instead of sequence position. 30 lines of PyTorch, 2% overhead, 1.25x compute savings.

Biggest gains come from questioning the stuff that seems too simple to be wrong.
