---
layout: post
title:  "from residual connections to attention residuals"
date:   2026-03-18 18:00:00 +0530
categories: tech
tokens: "~9k"
math: true
description: "The residual connection solved deep learning in 2015. Ten years later, Moonshot AI noticed it's been sabotaging deep networks the whole time. The fix is 30 lines of PyTorch."
---

> **TL;DR:** ResNet's skip connections fixed vanishing gradients in 2015 by having layers learn the residual F(x) = H(x) - x instead of the full mapping. In deep transformers, that same additive pattern dilutes the attention signal as the network scales. Moonshot AI's fix rescales the residual branch - a small code change with a measurable gain on deep models.

Quick context about resnets: neural networks learn by [backpropagating](/tech/2025/10/25/nn.html) gradients through layers. The deeper the network, the more times those gradients get multiplied by small numbers (sigmoid derivatives, weight matrices). After enough layers, [gradients shrink to zero](/tech/2026/01/21/math.html) - early layers stop learning entirely. This is the vanishing gradient problem, and it's why stacking more layers didn't work for a long time.

ResNet fixed this in 2015. Then everyone moved on. Ten years later, Moonshot AI looked at the fix.

---

## deeper should be better

Stack more layers. Learn more complex features. Get better results. That's the intuition - and it's wrong.

In 2015, He et al. trained plain (no skip connections) convolutional networks of increasing depth on CIFAR-10. The expectation: a 56-layer network should beat a 20-layer network because it has strictly more capacity. A 56-layer net can represent everything a 20-layer net can (set the extra 36 layers to identity) plus more.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/degradation_poster.jpg">
  <source src="/assets/images/attn_res/degradation.mp4" type="video/mp4">
</video>

| Model | Test Error |
|-------|-----------|
| Plain-20 | 9.26% |
| Plain-32 | 10.00% |
| Plain-44 | 11.22% |
| Plain-56 | 13.58% |

More layers, *worse* accuracy. And this isn't overfitting - the training error is also higher for deeper networks. Gradients vanish as they backpropagate through dozens of layers, and the optimizer can't find a good solution.

![Side by side comparison of plain vs residual networks during training](/assets/images/attn_res/side_by_side.png)

The degradation problem: theoretical capacity is there, but the optimizer can't reach it.

---

## the fix: skip connections

The insight from He et al. (2015): learn the *residual* $F(x) = H(x) - x$ rather than the full mapping $H(x)$ directly, then reconstruct the output as:

$$H(x) = F(x) + x$$

That's a skip connection. The input $x$ bypasses the convolutional layers and gets added directly to the output.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/skip_connection_poster.jpg">
  <source src="/assets/images/attn_res/skip_connection.mp4" type="video/mp4">
</video>

If a layer is useless, the network learns $F(x) = 0$ and the data passes through unchanged. Adding more layers can never make things worse - the worst case is identity. The optimizer starts from "do nothing" and improves from there.

```python
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)       # the skip connection
        return F.relu(out)
```

That `out += self.shortcut(x)` is the entire idea. One line. When dimensions match, `self.shortcut` is identity (literally `nn.Sequential()` with no ops). When they don't (downsampling), a 1x1 conv handles the projection.

---

## proof it works

Same architectures, same training setup - just add skip connections:

![Plain vs residual network accuracy comparison](/assets/images/attn_res/plain_vs_residual.png)

| Model | Test Error |
|-------|-----------|
| Plain-20 | 9.26% |
| ResNet-20 | 8.27% |
| Plain-56 | 13.58% |
| ResNet-56 | 6.41% |

Plain-56 was the worst model. ResNet-56 is the best. Skip connections fix the degradation problem and let deeper networks use their capacity. Within a year, He et al. won ImageNet with a 152-layer ResNet. That was unthinkable before skip connections.

The full implementation with training code, plots, and all ResNet variants: [deep-residual-learning-pytorch](https://github.com/samitmohan/deep-residual-learning-pytorch)

`x = x + F(x)` became the default wiring for ResNets, transformers, and diffusion models alike. Nobody touched it for ten years.

---

## the one line nobody questioned

```python
x = x + self.attn(self.norm1(x))  # this is in every LLM ever
```

But the weight on every layer's contribution is exactly **1.0**. Layer 1 gets the same importance as layer 64. The initial embedding gets the same weight as the final attention output. That's kinda dumb if you think about it.

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

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/norm_growth_poster.jpg">
  <source src="/assets/images/attn_res/norm_growth.mp4" type="video/mp4">
</video>

10,000x growth. No single layer is doing anything extreme - the problem is 64 outputs dumped into a running sum with no normalization.

### 2. information dilution

> imagine a meeting with 64 people. everyone speaks for exactly one minute. at the end you write a summary but you're forced to weight everyone equally.

That's standard residuals. After 64 layers, each layer contributes ~1.5% of the final hidden state. Layer 47 found a critical pattern between two tokens. Same weight as layer 3, which just copied the embedding.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/dilution_poster.jpg">
  <source src="/assets/images/attn_res/dilution.mp4" type="video/mp4">
</video>

The deeper your network, the less any individual layer matters. A 128-layer model gives each layer half the influence of a 64-layer model.

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

Prenorm should fix this - it normalizes input before each sub-layer:

```python
x = x + self.attn(self.norm1(x))  # norm bounds the input
```

The attention output is bounded. But you're adding that bounded output to `x`, which is the ever-growing running sum. Each new layer's signal is a smaller and smaller fraction of the total.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/prenorm_poster.jpg">
  <source src="/assets/images/attn_res/prenorm.mp4" type="video/mp4">
</video>

Last few layers of a 64-layer prenorm transformer contribute almost nothing. Bounded output, unbounded accumulator.

PreNorm solved gradient flow but not dilution. It made dilution worse.

---

## the fix: attention over depth

Fix: let the network **learn** the weights. Use attention - but instead of attending over sequence positions, attend over **depth**. Each block boundary looks at all previous checkpoint outputs and decides which ones matter.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/weights_comparison_poster.jpg">
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

RMSNorm ensures scores depend on the *direction* of each layer's output, not its magnitude. Without it, deeper layers with bigger norms would dominate the softmax.

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/depth_attn_poster.jpg">
  <source src="/assets/images/attn_res/depth_attn.mp4" type="video/mp4">
</video>

$w_l$ is **zero-initialized**. When $w = 0$, all dot products are 0, softmax gives uniform weights. Start from democracy, learn to specialize. Same trick as LoRA zero-init.

---

## full attnres vs block attnres

**Full version**: every layer attends over all previous outputs. $O(L^2 d)$ compute, $O(Ld)$ memory. Clean but expensive.

```python
# Full AttnRes: depth attention at EVERY layer
checkpoints = [x]
for i, layer in enumerate(layers):
    x = depth_attns[i](checkpoints)  # attend over all previous outputs
    x = layer(x)
    checkpoints.append(x)            # list grows: 1, 2, 3, ... L
```

Layer 32 attends over 32 checkpoints. Layer 64 attends over 64. The checkpoint list grows linearly, and each attention call scales with its length.

**Block version** (what you actually use): partition 64 layers into blocks of 8. Standard residuals within blocks, depth attention at boundaries only.

- Layers 1-8: standard `x = x + Layer(x)`, output $h_8$
- **Boundary**: depth attention over $\{h_0, h_8\}$ -> $h_8'$
- Layers 9-16: standard residuals from $h_8'$, output $h_{16}$
- **Boundary**: depth attention over $\{h_0, h_8', h_{16}\}$ -> $h_{16}'$
- ...repeat

<video autoplay loop muted playsinline preload="metadata" poster="/assets/images/attn_res/block_architecture_poster.jpg">
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

**1.25x compute savings** to match baseline quality. The +7.5 on GPQA-Diamond is wild for 30 lines of code.

Caveat: these are Moonshot's internal runs at massive scale. [Code is open source](https://github.com/MoonshotAI/Attention-Residuals) so anyone can try it, but reproducing exact numbers requires their data/compute.

---

## when this doesn't work

[Ziming Liu](https://kindxiaoming.github.io/blog/2026/attention-residual/) (the KAN guy) makes a good no-free-lunch argument: if AttnRes wins on language modeling, it has to lose somewhere else.

He constructs a simple experiment - interpolate between a structured linear task and a random memorization task, test both methods at different depths. The result:

- **Structured tasks** (patterns, relationships): AttnRes wins, and wins harder as depth increases.
- **Memorization tasks** (random input-output mappings): standard residuals win. AttnRes actually hurts.

Why? The zero-init trick that makes AttnRes safe also makes it dumb at the start. When $w = 0$, all weights are uniform: $h_l = \frac{1}{l}\left(h_0 + \sum_{i=1}^{l-1} v_i\right)$. That's pure averaging. The model has to learn its way out of this averaging regime before it can do anything useful. For memorization - where you need every layer to contribute maximally different signal - that averaging phase is a tax you never fully recover from.

The trade-off is **stability vs expressivity**. Softmax bounds the weights (they sum to 1), which stabilizes training but caps how much any single layer can contribute. Standard residuals let layers contribute unbounded signal - messier gradients, but more raw capacity for brute-force memorization.

At 4 blocks the methods are roughly equivalent. At 30 blocks the gap widens in both directions. Depth amplifies whatever advantage each method has on its preferred task type.

Language modeling is mostly structured (syntax, semantics, reasoning patterns), which is why AttnRes shows consistent gains on LLM benchmarks. But if your task is closer to lookup tables than language - don't assume this is a free upgrade.

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

## what else are we not questioning?

The residual weight was 1.0 for a decade. Softmax temperature is always 1/sqrt(d). Positional encodings are always added, never concatenated. Layer count is always uniform across the model. Somewhere in there is another 30-line patch worth +7.5 on GPQA. These frontier labs and their stanford smartass kids are looking at this in a much smarter way than I do.

---

## references

- [Attention Residuals](https://arxiv.org/abs/2603.15031) - Moonshot AI (Kimi team)
- [When does Kimi's Attention Residuals work?](https://kindxiaoming.github.io/blog/2026/attention-residual/) - Ziming Liu's no-free-lunch analysis
- [Attention Residuals Code](https://github.com/MoonshotAI/Attention-Residuals)
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [My CIFAR-10 ResNet implementation](https://github.com/samitmohan/deep-residual-learning-pytorch)
- [Vanishing gradients explained](/tech/2026/01/21/math.html) - from my math post
- [Backpropagation from scratch](/tech/2025/10/25/nn.html) - from my neural networks post
- [Transformers](/tech/2025/11/27/ai.html) - attention, residuals in the transformer block
- [AttnRes notebook](https://colab.research.google.com/drive/1FGVHFRY2ShFeQWPXTW30e_K-caW1_l-S) - standalone implementation with weight heatmap visualizations, by [Arjun](https://www.k-a.in/AttnRes.html)
