---
layout: post
title: "inference engineering"
date: 2026-05-02 00:00:00 +0530
categories: [tech]
tokens: "~15k"
math: true
description: "your GPU is mostly idle during text generation. the entire inference stack exists to fix that."
---

<style>
:root {
  --inf-bg: var(--color-background-secondary, #1a1a2e);
  --inf-border: var(--color-border, #2a2a4a);
  --inf-text: var(--color-text-primary, #e8e8e8);
  --inf-muted: var(--color-text-secondary, #888);
  --inf-blue: #4f9ef8;
  --inf-green: #4caf78;
  --inf-red: #f87171;
  --inf-orange: #f08c4b;
  --inf-purple: #a78bfa;
  --inf-yellow: #fbbf24;
  --inf-teal: #2dd4bf;
}
.inf-card { background: var(--inf-bg); border: 1px solid var(--inf-border); border-radius: 10px; padding: 16px 20px; margin: 1.2rem 0; font-size: 14px; color: var(--inf-muted); }
.inf-card strong { color: var(--inf-text); }
.inf-highlight { border-left: 3px solid var(--inf-orange); padding: 10px 14px; margin: 1rem 0; font-size: 13px; color: var(--inf-muted); background: var(--inf-bg); border-radius: 0 5px 5px 0; }
.inf-highlight strong { color: var(--inf-orange); }
.inf-insight { border-left: 3px solid var(--inf-blue); padding: 10px 14px; margin: 1rem 0; font-size: 13px; color: var(--inf-muted); background: var(--inf-bg); border-radius: 0 5px 5px 0; }
.inf-insight strong { color: var(--inf-blue); }
.inf-gloss { border-left: 3px solid var(--inf-purple); padding: 10px 14px; margin: 1rem 0; font-size: 13px; color: var(--inf-muted); background: var(--inf-bg); border-radius: 0 5px 5px 0; }
.inf-gloss strong { color: var(--inf-purple); }
.diag { margin: 1.8rem 0; text-align: center; }
.diag svg { max-width: 720px; width: 100%; }
.diag-caption { font-size: 12px; color: var(--inf-muted); margin-top: 6px; }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>

> **TL;DR:** Training teaches a model to think. Inference runs it thousands of times per second under a GPU budget. The whole stack exists to fight one problem: during token-by-token generation, your GPU is mostly idle waiting for memory. Every optimization below is a different way of fixing that.

---

## 1. what inference actually is

You finished training. You have a 16 GB blob of **weights** sitting on disk. A user types a prompt. You need to:

- load the weights into GPU memory
- run the prompt through the model to get a probability distribution over the next token
- pick one token, append it, repeat until done

That last loop is the painful part. Each output token = another full pass through the model. A 100-token reply = 100 sequential forward passes. Token N+1 depends on token N, so you can't parallelize this.

<div class="inf-insight">
<strong>analogy:</strong> The model is a 16 GB lookup table. To answer one question, the GPU reads every entry. To answer the next question, it reads every entry again. Thousands of cores sit idle waiting for the next chunk of data to arrive from memory. Inference engineering = keeping those cores busy.
</div>

### the iron triangle

Three things you trade off:

- **Latency** - how fast one user sees output
- **Throughput** - how many users per second per GPU
- **Cost** - $/million tokens

At `batch=1`, an H100 pushes ~200 tok/s from an 8B model. At `batch=128`: ~25,000 tok/s. Same GPU, same hourly rate. Per-token cost drops 128x. But more batching = each user waits longer.

<div class="inf-card">
<strong>The serving game:</strong> pack as many concurrent requests onto each GPU as possible without blowing latency past your SLO (e.g., "P99 TTFT under 500ms"). Every technique below either lets you pack more requests, or makes each one faster. That is the whole idea.
</div>

---

## 2. GPU 101 - the hardware vocab

You don't need to understand GPU internals. You need a few terms because every optimization below is just moving data between different kinds of memory. Skim this and refer back.

### VRAM and HBM - same thing, two names

Your CPU has RAM. The GPU has its own separate memory called **VRAM**. On AI GPUs, VRAM uses a technology called **HBM** (High Bandwidth Memory). "HBM" and "VRAM" mean the same thing in this post - the GPU's main memory pool.

- H100 has **80 GB** of HBM
- Separate from CPU RAM. Data copies over PCIe (slow)
- HBM bandwidth: ~3 TB/sec on H100. This number bottlenecks decode.

### SRAM - the GPU's tiny on-chip cache

Inside the GPU chip: tiny pools of much faster memory called **SRAM** (shared memory / L1 cache). Each compute unit has ~256 KB.

- ~6x faster than HBM (~19 TB/s vs ~3 TB/s)
- ~2400x smaller (256 KB vs 80 GB)
- FlashAttention's whole trick: do as much work as possible in SRAM before touching HBM

<div class="diag">
<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg">
  <text x="360" y="22" text-anchor="middle" font-family="monospace" font-size="16" font-weight="700" fill="var(--inf-text)">GPU memory hierarchy - speed vs size</text>
  <rect x="280" y="40" width="160" height="40" rx="4" fill="var(--inf-green)" opacity="0.5" stroke="var(--inf-green)" stroke-width="2"/>
  <text x="360" y="65" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-text)" font-weight="700">SRAM (on-chip)</text>
  <text x="475" y="55" font-family="monospace" font-size="12" fill="var(--inf-green)">256 KB / SM</text>
  <text x="475" y="73" font-family="monospace" font-size="12" fill="var(--inf-green)">~19 TB/s</text>
  <text x="200" y="55" text-anchor="end" font-family="monospace" font-size="11" fill="var(--inf-muted)">tiny</text>
  <text x="200" y="73" text-anchor="end" font-family="monospace" font-size="11" fill="var(--inf-green)" font-weight="700">FASTEST</text>
  <rect x="240" y="90" width="240" height="40" rx="4" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)" stroke-width="2"/>
  <text x="360" y="115" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-text)" font-weight="700">L2 cache</text>
  <text x="495" y="105" font-family="monospace" font-size="12" fill="var(--inf-muted)">~50 MB total</text>
  <text x="495" y="123" font-family="monospace" font-size="12" fill="var(--inf-muted)">~12 TB/s</text>
  <rect x="100" y="140" width="520" height="50" rx="4" fill="var(--inf-orange)" opacity="0.3" stroke="var(--inf-orange)" stroke-width="2"/>
  <text x="360" y="165" text-anchor="middle" font-family="monospace" font-size="14" fill="var(--inf-text)" font-weight="700">HBM (= VRAM) - the GPU's main memory</text>
  <text x="360" y="183" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">model weights live here. KV cache lives here.</text>
  <text x="635" y="155" font-family="monospace" font-size="12" fill="var(--inf-muted)">80 GB</text>
  <text x="635" y="173" font-family="monospace" font-size="12" fill="var(--inf-muted)">~3 TB/s</text>
  <text x="80" y="155" text-anchor="end" font-family="monospace" font-size="11" fill="var(--inf-muted)">huge</text>
  <text x="80" y="173" text-anchor="end" font-family="monospace" font-size="11" fill="var(--inf-red)" font-weight="700">SLOW</text>
  <rect x="40" y="200" width="640" height="35" rx="4" fill="none" stroke="var(--inf-muted)" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="360" y="222" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-muted)">CPU RAM - over PCIe (~30 GB/s, way too slow for inference)</text>
</svg>
<div class="diag-caption">Data flows up this hierarchy. The bottleneck in inference is the HBM bandwidth, not compute.</div>
</div>

### memory bandwidth - the number that matters

How fast data moves between memory and compute. On an H100:

- Compute: ~1000 TFLOPS
- HBM bandwidth: ~3 TB/sec

That ratio (~300 ops per byte) is everything. If your workload reads 1 byte and does 10 ops with it, you waste 290 op-cycles waiting for the next byte. That's decode. Welcome to memory-bound.

### tensor cores

Specialized units that do small matrix multiplications fast. Busy during prefill (compute-bound), idle during decode (memory-bound).

### GPU names you'll see

| Name | Generation | VRAM | Typical use |
|------|-----------|------|-------------|
| H100 / H200 | Hopper (NVIDIA) | 80 / 141 GB | Current flagship for inference |
| B200 | Blackwell | 192 GB | Newer, faster, expensive |
| A100 | Ampere | 40 / 80 GB | Previous gen, still common |
| RTX 4090 / 5090 | Consumer | 24 / 32 GB | Local / hobby inference |

<div class="inf-card">
<strong>HBM = the GPU's main memory.</strong> It's big but slow.<br>
<strong>SRAM = the GPU's on-chip cache.</strong> It's tiny but fast.<br>
<strong>Memory bandwidth = how fast data moves between them.</strong> When this is the bottleneck, your GPU sits idle.
</div>

---

## 3. prefill vs decode

Two phases, completely different bottlenecks. This split is the single most important mental model in this post.

<div class="diag">
<svg viewBox="0 0 800 230" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="inf-arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="var(--inf-muted)"/></marker>
  </defs>
  <rect x="20" y="20" width="340" height="190" rx="10" fill="var(--inf-green)" opacity="0.12" stroke="var(--inf-green)" stroke-width="2.5"/>
  <text x="195" y="50" text-anchor="middle" font-family="monospace" font-size="20" font-weight="700" fill="var(--inf-green)">PREFILL</text>
  <text x="195" y="75" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-muted)">process whole prompt at once</text>
  <rect x="80" y="95" width="80" height="45" rx="4" fill="var(--inf-green)" opacity="0.3"/>
  <text x="120" y="122" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">prompt</text>
  <text x="180" y="122" text-anchor="middle" font-family="monospace" font-size="18" fill="var(--inf-text)">x</text>
  <rect x="200" y="95" width="80" height="45" rx="4" fill="var(--inf-green)" opacity="0.3"/>
  <text x="240" y="122" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">weights</text>
  <text x="195" y="165" text-anchor="middle" font-family="monospace" font-size="14" fill="var(--inf-green)" font-weight="700">COMPUTE-BOUND</text>
  <text x="195" y="185" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">tensor cores saturated</text>
  <text x="195" y="200" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">drives TTFT</text>

  <line x1="370" y1="115" x2="425" y2="115" stroke="var(--inf-orange)" stroke-width="2" marker-end="url(#inf-arrow)"/>
  <text x="398" y="100" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-orange)" font-weight="700">KV cache</text>
  <text x="398" y="137" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">built here</text>

  <rect x="440" y="20" width="340" height="190" rx="10" fill="var(--inf-red)" opacity="0.12" stroke="var(--inf-red)" stroke-width="2.5"/>
  <text x="610" y="50" text-anchor="middle" font-family="monospace" font-size="20" font-weight="700" fill="var(--inf-red)">DECODE</text>
  <text x="610" y="75" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-muted)">one token at a time, in a loop</text>
  <rect x="500" y="100" width="24" height="35" rx="3" fill="var(--inf-red)" opacity="0.3"/>
  <text x="512" y="122" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">tok</text>
  <text x="542" y="122" text-anchor="middle" font-family="monospace" font-size="18" fill="var(--inf-text)">x</text>
  <rect x="560" y="95" width="100" height="45" rx="4" fill="var(--inf-red)" opacity="0.3"/>
  <text x="610" y="122" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">whole model</text>
  <text x="610" y="165" text-anchor="middle" font-family="monospace" font-size="14" fill="var(--inf-red)" font-weight="700">MEMORY-BOUND</text>
  <text x="610" y="185" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">GPU idle, waiting on HBM</text>
  <text x="610" y="200" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">drives TPOT</text>
</svg>
<div class="diag-caption">Prefill: matrix-matrix multiply (compute-bound). Decode: vector-matrix multiply (memory-bound).</div>
</div>

### prefill

Your prompt has 500 tokens. The model processes all 500 at once - one big matrix (prompt embeddings) times another big matrix (weights). Tensor cores light up. Every multiply-add unit does useful work. This is what GPUs are built for.

**Compute-bound.** Limited by FLOPS. Determines **TTFT** - time to first token. Long prompt = long prefill = user waits.

### decode

Now you're generating. You have one token. You multiply its hidden state (a tiny vector) by the entire 16 GB of model weights to predict the next token.

The math is trivial - a vector-matrix multiply. The pain is memory: the GPU reads all 16 GB from HBM for that one tiny computation. Then does it again for the next token. And again.

<div class="inf-insight">
<strong>why memory-bound?</strong> The ~300 ops-per-byte ratio from section 2. Decode does far fewer than 300 ops per byte read. The GPU sits idle waiting for memory. That is the entire problem.
</div>

**Memory-bandwidth-bound.** More compute doesn't help. Only reading less from memory helps. Determines **TPOT** - time per output token.

<div class="inf-card" style="border-color: var(--inf-red);">
<strong>the central insight:</strong> Prefill and decode have different bottlenecks. What helps prefill (more FLOPS) doesn't help decode. What helps decode (less memory traffic, bigger batches) doesn't help prefill. Production systems treat them as separate problems.
</div>

---

## 4. the metrics that matter

Four numbers. If you can't recite these about your deployment, you don't understand it.

**TTFT - Time To First Token.** Request arrival to first character visible. Driven by prefill + queuing. Anything > 1s feels laggy. Target < 500ms P99 for chat.

**TPOT - Time Per Output Token.** Gap between consecutive tokens during decode. Driven by HBM bandwidth + batch size. < 50ms feels smooth (~20 tok/s). Agent loops can tolerate < 100ms.

**Goodput.** Requests/second that meet your SLO. Raw throughput is misleading - 1000 req/s means nothing if half miss the latency target.

**MFU - Model FLOPS Utilization.** Fraction of peak FLOPS you use. Training hits 40-60%. Inference hits 10-30% - normal, because decode is memory-bound.

<div class="inf-card" style="border-color: var(--inf-red);">
<strong>averages lie.</strong> One slow request in ten makes an app feel broken. Track P50, P95, P99. Never the mean. P99 TTFT in a chat app with a million DAU = 10,000 unhappy users.
</div>

```bash
# measuring with genai-perf (production load test)
$ genai-perf profile \
    --model Qwen/Qwen3-8B-AWQ \
    --endpoint-type chat \
    --url localhost:8000 \
    --concurrency 16 \
    --input-tokens-mean 512 \
    --output-tokens-mean 128

# reports P50/P90/P99 for TTFT, TPOT, throughput, goodput
# run before AND after each optimization
```

---

## 5. the KV cache

The central data structure of LLM inference. If you only learn one thing from this post, make it this.

### why it exists

<div class="inf-gloss">
<strong>attention, briefly:</strong> Transformers work via attention. For every token, the model computes three vectors - Query (Q), Key (K), and Value (V). Don't worry about the math. What matters:
</div>

To generate token N, attention looks at the K and V of every previous token (1 through N-1).

Naively, you'd recompute K and V for every previous token at every step. Generating the 100th token = redoing work for tokens 1-99. 99% wasted compute.

<div class="inf-insight">
<strong>the fix:</strong> K and V for a token never change once computed. Compute them once, cache them in HBM. That cache - one entry per token, per layer, per attention head - is the <strong>KV cache</strong>. Just memoization.
</div>

### why it dominates memory

The KV cache grows with every token. The formula:

$$\text{KV size} = 2 \times B \times L \times H_{kv} \times d \times N \times \text{bytes}$$

B=batch, L=layers, $$H_{kv}$$=KV heads, d=head dim, N=seq len. The **2** is for storing both K and V.

Plug in Llama 70B (L=80, $$H_{kv}$$=8, d=128) at FP16, one user, 4K context:

$$2 \times 1 \times 80 \times 8 \times 128 \times 4096 \times 2 = \textbf{10.7 GB}$$

That's per user. Bump to batch=32: **342 GB**. Llama 70B itself is only 140 GB.

<div class="diag">
<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg">
  <text x="360" y="22" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-text)">VRAM usage on H100 (80GB) - Llama 70B</text>
  <text x="20" y="55" font-family="monospace" font-size="12" fill="var(--inf-muted)">model weights (140GB needs 2 GPUs)</text>
  <rect x="20" y="63" width="580" height="24" fill="var(--inf-blue)" opacity="0.5" stroke="var(--inf-blue)" stroke-width="1.5" rx="3"/>
  <text x="310" y="80" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">140 GB (constant)</text>
  <text x="20" y="112" font-family="monospace" font-size="12" fill="var(--inf-muted)">KV cache, batch=1, 4K context</text>
  <rect x="20" y="120" width="44" height="24" fill="var(--inf-orange)" opacity="0.6" stroke="var(--inf-orange)" stroke-width="1.5" rx="3"/>
  <text x="78" y="137" font-family="monospace" font-size="12" fill="var(--inf-muted)">10.7 GB</text>
  <text x="20" y="170" font-family="monospace" font-size="12" fill="var(--inf-muted)">KV cache, batch=32, 4K context</text>
  <rect x="20" y="178" width="580" height="24" fill="var(--inf-red)" opacity="0.6" stroke="var(--inf-red)" stroke-width="1.5" rx="3"/>
  <text x="310" y="195" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)" font-weight="700">342 GB - bigger than the model itself</text>
</svg>
<div class="diag-caption">The KV cache, not the model weights, is the memory bottleneck of LLM serving.</div>
</div>

<div class="inf-card">
<strong>The plot twist:</strong> the KV cache, not the model weights, is the memory bottleneck. Every "fit more requests on this GPU" technique is really just "make the KV cache smaller."
</div>

### how to fight it

Three levers, in order of frequency:

- **PagedAttention** - eliminates wasted reservation (next section)
- **Prefix caching** - reuse cache across requests with shared prefixes
- **KV cache quantization** - store K, V in INT8/FP8 instead of FP16. Halves memory. < 0.5% quality loss.

```python
# turn on KV quantization in vLLM
llm = LLM(
    model="meta-llama/Llama-3-70B",
    kv_cache_dtype="fp8",    # half the KV memory
    gpu_memory_utilization=0.90,
)
```

### MHA / MQA / GQA / MLA

You don't choose this - the model architect did. But it explains why two similar-sized models can have 10x different KV cache sizes:

| Variant | What it shares | Used by |
|---------|---------------|---------|
| **MHA** | Nothing - each head has its own K, V (original transformer) | GPT-2 |
| **MQA** | All heads share one K, V pair. Aggressive savings. | PaLM |
| **GQA** | Heads share K, V in groups of 4-8. Sweet spot. | Llama 2/3, Mistral, Qwen |
| **MLA** | Projects K, V into a tiny latent space, reconstructs at attention time. ~10x compression. | DeepSeek V2/V3 |

This is why Llama 3 70B and DeepSeek V3 (671B) have similar serving costs. DeepSeek's MLA + MoE keeps the active KV cache tiny despite being 10x larger.

---

## 6. FlashAttention

You'll never write this yourself. But it's why long-context inference works at all.

### the problem

Standard attention computes **S = Q x K^T**. That's an N x N matrix. At N=4K: 64 MB. At N=128K: **32 GB** - bigger than most GPUs.

The naive algorithm writes the full matrix to HBM, reads it back, applies softmax, writes again, reads again, multiplies by V. Four HBM round trips for a matrix that exists only to be immediately consumed.

<div class="diag">
<svg viewBox="0 0 760 230" xmlns="http://www.w3.org/2000/svg">
  <text x="170" y="20" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-red)">standard attention</text>
  <rect x="70" y="35" width="200" height="180" rx="6" fill="var(--inf-red)" opacity="0.1" stroke="var(--inf-red)" stroke-width="2"/>
  <line x1="70" y1="80" x2="270" y2="80" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="70" y1="125" x2="270" y2="125" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="70" y1="170" x2="270" y2="170" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="120" y1="35" x2="120" y2="215" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="170" y1="35" x2="170" y2="215" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="220" y1="35" x2="220" y2="215" stroke="var(--inf-red)" opacity="0.2"/>
  <text x="170" y="118" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-red)">full N x N matrix</text>
  <text x="170" y="136" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-red)">in HBM (slow)</text>
  <text x="170" y="154" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-red)">O(N^2) memory</text>

  <text x="380" y="125" text-anchor="middle" font-family="monospace" font-size="24" fill="var(--inf-muted)" font-weight="700">vs</text>

  <text x="580" y="20" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-green)">FlashAttention</text>
  <rect x="480" y="35" width="200" height="180" rx="6" fill="none" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="5,4"/>
  <rect x="490" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="535" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="580" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="625" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="490" y="90" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="535" y="90" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="580" y="90" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="625" y="90" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="490" y="135" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="535" y="135" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="580" y="135" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="625" y="135" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="490" y="180" width="40" height="22" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="535" y="180" width="40" height="22" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="580" y="180" width="40" height="22" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="625" y="180" width="40" height="22" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <text x="580" y="125" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-green)">tiled blocks</text>
  <text x="580" y="143" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-green)">in SRAM</text>
</svg>
<div class="diag-caption">Left: standard attention materializes N x N in HBM. Right: FlashAttention tiles it in SRAM, block by block.</div>
</div>

### the fix (Dao et al., 2022)

Never build the full matrix. Instead:

- **Tile** Q, K, V into blocks that fit in SRAM
- For each Q block, iterate over K, V blocks. Compute attention scores in SRAM, never writing the intermediate matrix to HBM
- **Online softmax** keeps running statistics so softmax computes correctly as blocks arrive. Exact result, not approximate

<div class="inf-card" style="border-color: var(--inf-green);">
<strong>net effect:</strong> O(N^2) memory drops to O(N). Same exact output. Way fewer memory trips. At long context, this is "works" vs "OOM crash."
</div>

Automatically enabled in vLLM, SGLang, and PyTorch's `scaled_dot_product_attention`. You don't configure it.

---

## 7. vLLM's superpowers

<div class="inf-gloss">
<strong>what's vLLM?</strong> The most popular open-source LLM serving engine. Nginx for language models - point it at a model, get a high-throughput inference server. SGLang and TGI are alternatives. All bundle the four optimizations below.
</div>

Together these four give 5-10x throughput over a naive HuggingFace `.generate()` loop. All configurable, and at scale you'll tune them.

### 7.1 - PagedAttention (OS virtual memory for the KV cache)

Naive KV cache allocation: reserve max possible context length per request. Model supports 128K context? Reserve gigabytes. Even if the user sends "hi".

If you've taken an OS class, you know this story:

<div class="diag">
<svg viewBox="0 0 760 180" xmlns="http://www.w3.org/2000/svg">
  <text x="150" y="18" text-anchor="middle" font-family="monospace" font-size="14" font-weight="700" fill="var(--inf-text)">logical (block table)</text>
  <rect x="20" y="30" width="260" height="35" rx="5" fill="none" stroke="var(--inf-blue)" stroke-width="1.5"/>
  <text x="35" y="52" font-family="monospace" font-size="12" fill="var(--inf-blue)">Req 1: B0 -> B3 -> B7 -> B9</text>
  <rect x="20" y="72" width="260" height="35" rx="5" fill="none" stroke="var(--inf-purple)" stroke-width="1.5"/>
  <text x="35" y="94" font-family="monospace" font-size="12" fill="var(--inf-purple)">Req 2: B0 -> B3 -> B5 -> B8</text>
  <text x="150" y="130" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-green)">B0, B3 shared (copy-on-write)</text>

  <line x1="295" y1="65" x2="365" y2="65" stroke="var(--inf-muted)" stroke-width="1.5" marker-end="url(#inf-arrow)"/>

  <text x="560" y="18" text-anchor="middle" font-family="monospace" font-size="14" font-weight="700" fill="var(--inf-text)">physical VRAM</text>
  <rect x="380" y="35" width="50" height="35" rx="4" fill="var(--inf-green)" opacity="0.5" stroke="var(--inf-green)"/><text x="405" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B0</text>
  <rect x="438" y="35" width="50" height="35" rx="4" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/><text x="463" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B1</text>
  <rect x="496" y="35" width="50" height="35" rx="4" fill="var(--inf-border)" opacity="0.5" stroke="var(--inf-muted)"/><text x="521" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">free</text>
  <rect x="554" y="35" width="50" height="35" rx="4" fill="var(--inf-green)" opacity="0.5" stroke="var(--inf-green)"/><text x="579" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B3</text>
  <rect x="612" y="35" width="50" height="35" rx="4" fill="var(--inf-purple)" opacity="0.3" stroke="var(--inf-purple)"/><text x="637" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B5</text>
  <rect x="670" y="35" width="50" height="35" rx="4" fill="var(--inf-border)" opacity="0.5" stroke="var(--inf-muted)"/><text x="695" y="57" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">free</text>

  <rect x="380" y="78" width="50" height="35" rx="4" fill="var(--inf-purple)" opacity="0.3" stroke="var(--inf-purple)"/><text x="405" y="100" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B8</text>
  <rect x="438" y="78" width="50" height="35" rx="4" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/><text x="463" y="100" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B7</text>
  <rect x="496" y="78" width="50" height="35" rx="4" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/><text x="521" y="100" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">B9</text>
  <rect x="554" y="78" width="50" height="35" rx="4" fill="var(--inf-border)" opacity="0.5" stroke="var(--inf-muted)"/><text x="579" y="100" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">free</text>

  <text x="560" y="140" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">scattered, not contiguous - near-zero waste</text>
</svg>
<div class="diag-caption">PagedAttention: a block table maps logical positions to scattered physical blocks. Shared prefixes use copy-on-write.</div>
</div>

- Split VRAM into fixed-size **blocks** (16 tokens each)
- Allocate on demand as the sequence grows
- Block table maps logical positions to scattered physical blocks (just like OS page tables)
- **Copy-on-write**: two requests share a system prompt? Same physical blocks. Fork only when they diverge.

This is vLLM's whole reason for existing. On by default. Explains why it packs 2-3x more requests per GPU than a naive setup.

### 7.2 - continuous batching

Static batching: collect N requests, pad to same length, process as one batch, return together. Short responses sit idle waiting for the longest. Massive waste.

<div class="diag">
<svg viewBox="0 0 720 270" xmlns="http://www.w3.org/2000/svg">
  <text x="20" y="18" font-family="monospace" font-size="14" fill="var(--inf-red)" font-weight="700">static batching (bad)</text>
  <rect x="20" y="28" width="280" height="20" rx="3" fill="var(--inf-blue)" opacity="0.6"/>
  <rect x="300" y="28" width="60" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <rect x="20" y="54" width="180" height="20" rx="3" fill="var(--inf-green)" opacity="0.6"/>
  <rect x="200" y="54" width="160" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <rect x="20" y="80" width="350" height="20" rx="3" fill="var(--inf-purple)" opacity="0.6"/>
  <rect x="20" y="106" width="120" height="20" rx="3" fill="var(--inf-orange)" opacity="0.6"/>
  <rect x="140" y="106" width="220" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <text x="420" y="55" font-family="monospace" font-size="12" fill="var(--inf-muted)">grey = idle (padding)</text>
  <text x="420" y="75" font-family="monospace" font-size="12" fill="var(--inf-muted)">all wait for slowest</text>

  <text x="20" y="160" font-family="monospace" font-size="14" fill="var(--inf-green)" font-weight="700">continuous batching (good)</text>
  <rect x="20" y="170" width="280" height="20" rx="3" fill="var(--inf-blue)" opacity="0.6"/>
  <rect x="300" y="170" width="100" height="20" rx="3" fill="var(--inf-teal)" opacity="0.7"/>
  <rect x="20" y="196" width="180" height="20" rx="3" fill="var(--inf-green)" opacity="0.6"/>
  <rect x="200" y="196" width="200" height="20" rx="3" fill="var(--inf-yellow)" opacity="0.7"/>
  <rect x="20" y="222" width="350" height="20" rx="3" fill="var(--inf-purple)" opacity="0.6"/>
  <rect x="20" y="248" width="120" height="20" rx="3" fill="var(--inf-orange)" opacity="0.6"/>
  <rect x="140" y="248" width="130" height="20" rx="3" fill="var(--inf-teal)" opacity="0.7"/>
  <rect x="270" y="248" width="100" height="20" rx="3" fill="var(--inf-red)" opacity="0.5"/>
  <text x="420" y="200" font-family="monospace" font-size="12" fill="var(--inf-muted)">new requests slot in</text>
  <text x="420" y="220" font-family="monospace" font-size="12" fill="var(--inf-muted)">when old ones finish</text>
  <text x="420" y="244" font-family="monospace" font-size="13" fill="var(--inf-green)" font-weight="700">2-5x throughput</text>
</svg>
<div class="diag-caption">Static batching wastes GPU cycles on padding. Continuous batching keeps the batch full.</div>
</div>

**Continuous batching**: at every decode step, check if any request finished. If so, evict it and slot in a new request from the queue. Batch stays full. GPU stays busy.

<div class="inf-insight">
<strong>SWE analogy:</strong> Static batching = thread-per-request server waiting for the slowest connection. Continuous batching = event loop. Admit new work as soon as a slot opens.
</div>

On by default. You don't configure it; you benefit from it.

### 7.3 - chunked prefill

Even with continuous batching: a long prompt enters the queue, its 500ms prefill stalls every decode-phase request in the batch. **Prefill piracy.** Latency spikes for everyone.

**Chunked prefill**: break long prefills into 512-token chunks, interleave decode steps between them. The long prefill still finishes. But no one else's TPOT spikes.

<div class="inf-insight">
<strong>SWE analogy:</strong> Preemptive scheduling. No single process starves others.
</div>

### 7.4 - prefix caching

Most production traffic shares prefixes: same system prompt, same few-shot examples, same RAG context. Without caching, every request recomputes KV for the shared prefix. Thousands of tokens of duplicate work.

**Automatic prefix caching**: store KV blocks keyed by token content. New requests with matching prefix reuse cached blocks, skip that portion of prefill. TTFT drops to near-zero for the shared part.

```python
# turn on all four in vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B-AWQ",
    quantization="awq",
    enable_prefix_caching=True,    # 7.4 - reuse shared prefixes
    enable_chunked_prefill=True,   # 7.3 - no more prefill piracy
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    # PagedAttention (7.1) and continuous batching (7.2) are always on
)
```

Three flag flips. ~5x over raw `.generate()`. These are table stakes.

---

## 8. quantization

The single biggest lever you have.

### why it works

Weights default to FP16/BF16 - 16 bits per number. A 70B model = 70 billion x 2 bytes = **140 GB**.

Quantization: store those numbers in fewer bits. INT4 = 4 bits = 35 GB for the same model. 4x less memory, 4x less bandwidth consumed during decode.

Decode is memory-bound. Read 4x less from HBM = up to 4x faster generation. That's it.

<div class="inf-insight">
<strong>won't quality crash?</strong> No - up to a point. Weights have redundancy. 8-bit loses < 1%. 4-bit loses ~5%. Below 4-bit, the model breaks.
</div>

<div class="inf-gloss">
<strong>how is quality measured?</strong> <strong>Perplexity</strong> - how "surprised" the model is by held-out text. Lower = better. 5% increase is invisible to users.
</div>

### number formats you'll meet

| Format | Bits | When to use |
|--------|------|-------------|
| FP32 | 32 | Training default. Don't use for inference. |
| BF16 | 16 | Inference default. Baseline. |
| FP8 | 8 | Hopper+ GPUs. Near-lossless. Zero effort. |
| INT8 (W8A8) | 8 | Weights AND activations in 8-bit. Faster prefill too. |
| INT4 (AWQ/GPTQ) | 4 | Biggest decode speedup. ~5% quality loss. |
| INT2/INT3 | <=3 | Model breaks. Research only. |

```python
# FP8 if you have a Hopper GPU
llm = LLM(
    model="meta-llama/Llama-3-70B",
    quantization="fp8",            # on-the-fly, no separate model
    kv_cache_dtype="fp8",         # bonus: KV in FP8 too
)
```

<div class="inf-card" style="border-color: var(--inf-green);">
<strong>first thing to try:</strong> Out of everything in this post, quantization gives the biggest single jump. Try it before anything else.
</div>

---

## 9. when one GPU isn't enough

Llama 70B at FP16 = 140 GB. H100 = 80 GB. Doesn't fit. Even at 4-bit (35 GB), production KV cache pushes past 80 GB. You need multiple GPUs.

### tensor parallelism (TP) - the one you'll use

Split each weight matrix across GPUs. Each GPU holds 1/N of every weight. At inference time, each computes its slice, then they exchange partial results via **AllReduce** (every GPU contributes its partial, ends up with the sum).

<div class="diag">
<svg viewBox="0 0 720 170" xmlns="http://www.w3.org/2000/svg">
  <text x="360" y="20" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-text)">Tensor Parallelism, TP=4</text>
  <text x="100" y="50" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-muted)">full weight matrix W</text>
  <rect x="40" y="58" width="120" height="70" fill="var(--inf-blue)" opacity="0.4" stroke="var(--inf-blue)" stroke-width="2" rx="4"/>
  <text x="100" y="98" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">[M x N]</text>

  <line x1="175" y1="93" x2="225" y2="93" stroke="var(--inf-muted)" stroke-width="2" marker-end="url(#inf-arrow)"/>

  <text x="400" y="50" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-muted)">split across 4 GPUs (column-wise)</text>
  <rect x="240" y="58" width="55" height="70" fill="var(--inf-green)" opacity="0.4" stroke="var(--inf-green)" stroke-width="2" rx="4"/>
  <text x="267" y="98" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">GPU 0</text>
  <rect x="300" y="58" width="55" height="70" fill="var(--inf-orange)" opacity="0.4" stroke="var(--inf-orange)" stroke-width="2" rx="4"/>
  <text x="327" y="98" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">GPU 1</text>
  <rect x="360" y="58" width="55" height="70" fill="var(--inf-purple)" opacity="0.4" stroke="var(--inf-purple)" stroke-width="2" rx="4"/>
  <text x="387" y="98" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">GPU 2</text>
  <rect x="420" y="58" width="55" height="70" fill="var(--inf-red)" opacity="0.4" stroke="var(--inf-red)" stroke-width="2" rx="4"/>
  <text x="447" y="98" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">GPU 3</text>

  <line x1="490" y1="93" x2="540" y2="93" stroke="var(--inf-red)" stroke-width="2" marker-end="url(#inf-arrow)"/>
  <text x="630" y="88" text-anchor="middle" font-family="monospace" font-size="16" fill="var(--inf-red)" font-weight="700">AllReduce</text>
  <text x="630" y="108" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">(every layer)</text>

  <text x="360" y="155" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">needs NVLink between GPUs (inside one server box)</text>
</svg>
<div class="diag-caption">TP splits weight matrices column-wise. An AllReduce synchronizes results on every layer.</div>
</div>

- AllReduce happens on every layer. Lots of communication.
- Needs **NVLink** (~900 GB/s, GPU-to-GPU within one box). InfiniBand across nodes is too slow (~50 GB/s).
- Max `TP=8` because that's how many H100s fit in one NVLink domain.

<div class="inf-card">
<strong>Rule of thumb:</strong> use the smallest TP that fits your model + KV cache. <code>TP=2</code> is faster than <code>TP=4</code> for the same model because there's half the AllReduce overhead.
</div>

```python
# vLLM with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,    # split across 4 GPUs in the same node
    gpu_memory_utilization=0.90,
)
```

### the other two (safe to skip)

**Pipeline parallelism (PP)** - split by layers. Point-to-point communication, works across nodes on InfiniBand. Catch: pipeline bubbles. Only matters for 400B+ models.

**Expert parallelism (EP)** - for MoE models. Distributes experts across GPUs via All-to-All. Skip unless deploying MoE.

Production combo: `TP=8` within a node + `PP=N` across nodes. For 70B or smaller: just `TP=2` or `TP=4`.

---

## 10. mixture of experts (MoE)

Standard transformers are **dense** - every token passes through every parameter. MoE replaces the dense feedforward layers with many smaller **expert** networks plus a learned **router**. Each token activates only a few (e.g., 8 of 128). Per-token compute stays low despite massive total parameters.

DeepSeek-V3: 671B total, 37B active per token. Quality of a 671B model, compute cost of a 37B model.

<div class="diag">
<svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg">
  <text x="360" y="20" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-text)">MoE: router picks 2 of 8 experts per token</text>

  <rect x="60" y="50" width="80" height="40" rx="4" fill="var(--inf-blue)" opacity="0.4" stroke="var(--inf-blue)" stroke-width="1.5"/>
  <text x="100" y="75" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">token</text>

  <line x1="145" y1="70" x2="210" y2="70" stroke="var(--inf-muted)" stroke-width="1.5" marker-end="url(#inf-arrow)"/>

  <rect x="215" y="45" width="80" height="50" rx="4" fill="var(--inf-orange)" opacity="0.4" stroke="var(--inf-orange)" stroke-width="1.5"/>
  <text x="255" y="67" text-anchor="middle" font-family="monospace" font-size="12" fill="var(--inf-text)">router</text>
  <text x="255" y="83" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">softmax</text>

  <line x1="300" y1="55" x2="370" y2="42" stroke="var(--inf-green)" stroke-width="2"/>
  <line x1="300" y1="80" x2="370" y2="102" stroke="var(--inf-green)" stroke-width="2"/>
  <line x1="300" y1="60" x2="370" y2="62" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>
  <line x1="300" y1="65" x2="370" y2="82" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>
  <line x1="300" y1="70" x2="370" y2="122" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>
  <line x1="300" y1="75" x2="370" y2="142" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>
  <line x1="300" y1="80" x2="370" y2="162" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>
  <line x1="300" y1="85" x2="370" y2="182" stroke="var(--inf-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.3"/>

  <rect x="375" y="30" width="90" height="28" rx="4" fill="var(--inf-green)" opacity="0.6" stroke="var(--inf-green)" stroke-width="1.5"/>
  <text x="420" y="49" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">Expert 1</text>
  <rect x="375" y="62" width="90" height="28" rx="4" fill="var(--inf-border)" opacity="0.4" stroke="var(--inf-muted)" stroke-width="1"/>
  <text x="420" y="81" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">Expert 2</text>
  <rect x="375" y="94" width="90" height="28" rx="4" fill="var(--inf-green)" opacity="0.6" stroke="var(--inf-green)" stroke-width="1.5"/>
  <text x="420" y="113" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">Expert 3</text>
  <rect x="375" y="126" width="90" height="28" rx="4" fill="var(--inf-border)" opacity="0.4" stroke="var(--inf-muted)" stroke-width="1"/>
  <text x="420" y="145" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">Expert 4</text>
  <rect x="375" y="158" width="90" height="28" rx="4" fill="var(--inf-border)" opacity="0.4" stroke="var(--inf-muted)" stroke-width="1"/>
  <text x="420" y="177" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">...</text>
  <rect x="375" y="190" width="90" height="28" rx="4" fill="var(--inf-border)" opacity="0.4" stroke="var(--inf-muted)" stroke-width="1"/>
  <text x="420" y="209" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">Expert 8</text>

  <text x="490" y="49" font-family="monospace" font-size="10" fill="var(--inf-green)">selected</text>
  <text x="490" y="81" font-family="monospace" font-size="10" fill="var(--inf-muted)">skipped</text>
  <text x="490" y="113" font-family="monospace" font-size="10" fill="var(--inf-green)">selected</text>

  <line x1="470" y1="44" x2="560" y2="80" stroke="var(--inf-green)" stroke-width="1.5"/>
  <line x1="470" y1="108" x2="560" y2="80" stroke="var(--inf-green)" stroke-width="1.5"/>

  <rect x="565" y="60" width="110" height="40" rx="4" fill="var(--inf-purple)" opacity="0.3" stroke="var(--inf-purple)" stroke-width="1.5"/>
  <text x="620" y="77" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">weighted sum</text>
  <text x="620" y="93" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">-> next layer</text>
</svg>
<div class="diag-caption">Each token routes to 2 of 8 experts. The rest of the experts sit idle for this token. Green = active, grey = skipped.</div>
</div>

### the inference tradeoff

You need the **full model in VRAM** even though each token uses ~5% of it. The router doesn't know which experts a token needs until runtime - every expert must be loaded. In batched inference, different requests hit different experts, so most parameters stay warm.

This is why MoE pairs with **expert parallelism** (section 9): distribute experts across GPUs via All-to-All instead of replicating. DeepSeek-V3 combines MoE + MLA (section 5) - keeps both KV cache and active compute small. That's how 671B achieves serving costs comparable to Llama 70B.

### when MoE wins

- **Quality per FLOP**: larger model's knowledge, fraction of the compute
- **Throughput**: fewer active parameters = less bandwidth consumed per token
- **Scaling**: grow total parameters without proportional per-token cost increase

### when MoE hurts

- **Memory**: full model must fit in VRAM regardless of sparsity. 671B at FP8 = still ~670 GB.
- **Communication**: All-to-All degrades across slow interconnects
- **Load balancing**: if the router favors certain experts, you get hotspots. Training penalizes this, but imbalance still shows up in production.

---

## 11. serving the model

vLLM is configured. Now your app talks to it. This is where SWE work lives.

### the one-line server

vLLM ships an OpenAI-compatible HTTP server:

```bash
$ vllm serve Qwen/Qwen3-8B-AWQ \
    --quantization awq \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --port 8000

# now you have an OpenAI-compatible endpoint at localhost:8000
```

"OpenAI-compatible" = any client library that talks to the OpenAI API talks to your vLLM server unchanged. Switch `base_url` and you're done. Drop-in replacement.

### request lifecycle

<div class="diag">
<svg viewBox="0 0 760 200" xmlns="http://www.w3.org/2000/svg">
  <text x="380" y="20" text-anchor="middle" font-family="monospace" font-size="15" font-weight="700" fill="var(--inf-text)">how a request flows through vLLM</text>

  <rect x="20" y="70" width="100" height="55" rx="8" fill="var(--inf-blue)" opacity="0.2" stroke="var(--inf-blue)" stroke-width="1.5"/>
  <text x="70" y="93" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-text)" font-weight="700">your app</text>
  <text x="70" y="112" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">openai SDK</text>

  <line x1="125" y1="97" x2="180" y2="97" stroke="var(--inf-muted)" stroke-width="1.5" marker-end="url(#inf-arrow)"/>
  <text x="153" y="85" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">HTTP POST</text>

  <rect x="185" y="70" width="100" height="55" rx="8" fill="var(--inf-orange)" opacity="0.2" stroke="var(--inf-orange)" stroke-width="1.5"/>
  <text x="235" y="93" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-text)" font-weight="700">queue</text>
  <text x="235" y="112" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">scheduler</text>

  <line x1="290" y1="97" x2="345" y2="97" stroke="var(--inf-muted)" stroke-width="1.5" marker-end="url(#inf-arrow)"/>

  <rect x="350" y="55" width="180" height="85" rx="8" fill="var(--inf-green)" opacity="0.15" stroke="var(--inf-green)" stroke-width="1.5"/>
  <text x="440" y="73" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-green)" font-weight="700">GPU batch</text>
  <rect x="365" y="82" width="150" height="12" rx="3" fill="var(--inf-blue)" opacity="0.5"/>
  <rect x="365" y="98" width="150" height="12" rx="3" fill="var(--inf-purple)" opacity="0.5"/>
  <rect x="365" y="114" width="150" height="12" rx="3" fill="var(--inf-orange)" opacity="0.5"/>
  <text x="440" y="138" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">continuous batching</text>

  <line x1="535" y1="97" x2="590" y2="97" stroke="var(--inf-red)" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#inf-arrow)"/>
  <text x="563" y="85" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-red)">SSE stream</text>

  <rect x="595" y="70" width="120" height="55" rx="8" fill="var(--inf-blue)" opacity="0.2" stroke="var(--inf-blue)" stroke-width="1.5"/>
  <text x="655" y="93" text-anchor="middle" font-family="monospace" font-size="13" fill="var(--inf-text)" font-weight="700">your app</text>
  <text x="655" y="112" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-muted)">tokens flow in</text>

  <text x="380" y="175" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">first token = TTFT. each next token = TPOT later.</text>
</svg>
</div>

### streaming

Wait for the full response = user stares at a spinner for 5+ seconds. Stream tokens via **SSE** (Server-Sent Events) instead - server keeps the connection open, pushes tokens as they generate. Text appears word-by-word like ChatGPT.

The SDK handles this with `stream=True`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",    # vLLM doesn't check, but SDK requires non-empty
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[{"role": "user", "content": "explain CAP theorem"}],
    stream=True,
    max_tokens=512,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### tuning

Two knobs:

- `--max-num-seqs` - max concurrent requests in batch. Higher = more throughput + more KV memory + longer TPOT. Start at 64.
- `--max-num-batched-tokens` - total tokens per scheduler step. Caps prefill-decode packing.

### scaling out

One vLLM process saturates one GPU (or TP group). More traffic = more replicas behind a load balancer. Use **least-pending-requests** routing, not round-robin - LLM requests have wildly varying durations.

<div class="inf-card" style="border-color: var(--inf-blue);">
<strong>shipping checklist:</strong><br>
- <code>/health</code> endpoint for the LB<br>
- <code>/metrics</code> (Prometheus format, built-in)<br>
- generous request timeouts (long context = long requests)<br>
- auth via API gateway in front, never on the model server
</div>

---

## 12. the deployment playbook

Latency bad. Throughput bad. Bill bad. Where to start? This order:

<div class="inf-card">
<strong>0. Profile first.</strong> Bottleneck might be tokenization, queuing, or network - not the model. <code>torch.profiler</code> or NSight Systems before touching anything.<br><br>
<strong>1. Switch to vLLM/SGLang.</strong> If you're on raw <code>.generate()</code>, this alone is 5-10x.<br><br>
<strong>2. Quantize.</strong> AWQ 4-bit or FP8. Biggest single jump.<br><br>
<strong>3. Prefix caching + chunked prefill.</strong> Free wins for shared system prompts.<br><br>
<strong>4. Right-size GPU count.</strong> Smallest TP that fits. <code>TP=2</code> beats <code>TP=4</code>.<br><br>
<strong>5. KV cache quantization.</strong> <code>kv_cache_dtype="fp8"</code> - doubles effective batch size.<br><br>
<strong>6. Speculative decoding.</strong> 2-3x for predictable outputs. Skip for creative generation.<br><br>
<strong>7. Disaggregate prefill/decode.</strong> Different GPU pools per phase. Worth it at 100+ GPUs.
</div>
