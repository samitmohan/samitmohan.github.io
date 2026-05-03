---
layout: post
title: "inference engineering"
date: 2026-05-02 00:00:00 +0530
categories: [tech]
tokens: "~30k"
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
.diag { margin: 1.8rem 0; text-align: center; }
.diag svg { max-width: 720px; width: 100%; }
.diag-caption { font-size: 12px; color: var(--inf-muted); margin-top: 6px; }
.inf-video { margin: 1.8rem 0; text-align: center; }
.inf-video video { max-width: 100%; border-radius: 10px; border: 1px solid var(--inf-border); }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>

> **TL;DR:** Training teaches a model to think. Inference makes it think on demand, thousands of times per second, under a GPU budget. The decode phase is memory-bound, and the entire inference stack exists to fix that.

I wrote about [what happens when you press submit on ChatGPT](/tech/2026/03/27/how-chatgpt-works.html) covering pretraining, alignment, and inference at a high level. This post goes deeper on the inference side. How do you take a trained model and serve it to millions of users without going bankrupt on GPU bills?

The answers involve OS-style virtual memory for attention caches, tiled matrix multiplication in on-chip SRAM, small models that guess ahead for big models, and turning 32-bit floats into 4-bit integers without destroying output quality.

---

## the iron triangle

At batch=1, a single H100 pushes 200 tokens per second through an 8B model. At batch=128: 25,000 tokens per second, same GPU, same hourly rate. Per-token cost drops 128x.

The constraint: 128 concurrent requests each carry a KV cache - the attention state from all previous tokens. At 4K context, that's 64GB of KV cache fighting for space alongside model weights. Memory runs out before compute does.

If you've scaled web services, this tension is familiar. Connection pools, request buffers, memory pressure - same dynamics, different hardware. Drag the batch size and watch:

<style>
.iron-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.iron-widget label{font-size:13px;display:block;margin-bottom:4px;color:var(--inf-muted);font-family:monospace}
.iron-widget input[type=range]{width:100%;margin:4px 0 16px}
.iron-grp{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--inf-muted);font-family:monospace;margin:12px 0 4px;opacity:0.7}
.iron-row{display:flex;align-items:center;gap:10px;margin:6px 0;font-family:monospace;font-size:13px}
.iron-name{width:90px;color:var(--inf-muted);text-align:right;flex-shrink:0;font-size:12px}
.iron-bar-outer{flex:1;height:16px;background:var(--inf-border);border-radius:8px;overflow:hidden}
.iron-bar-inner{height:100%;border-radius:8px;transition:width 0.3s,background 0.3s}
.iron-val{width:140px;color:var(--inf-text);font-weight:600;font-size:12px;flex-shrink:0}
.iron-note{font-size:12px;color:var(--inf-green);font-family:monospace;margin-top:10px;opacity:0.9}
.iron-insight{font-size:13px;margin-top:12px;padding:10px 14px;border-left:3px solid var(--inf-orange);background:var(--inf-bg);border-radius:0 5px 5px 0;font-family:monospace;color:var(--inf-muted);min-height:36px}
</style>
<div class="iron-widget">
  <label>Concurrent requests (batch size): <strong id="iron-batch">1</strong></label>
  <input type="range" min="1" max="256" value="1" id="iron-batch-slider" oninput="updateIron()">
  <div class="iron-grp">what improves</div>
  <div class="iron-row">
    <div class="iron-name">Throughput</div>
    <div class="iron-bar-outer"><div class="iron-bar-inner" id="iron-tput-bar"></div></div>
    <div class="iron-val" id="iron-tput-val"></div>
  </div>
  <div class="iron-row">
    <div class="iron-name">Cost / token</div>
    <div class="iron-bar-outer"><div class="iron-bar-inner" id="iron-cost-bar"></div></div>
    <div class="iron-val" id="iron-cost-val"></div>
  </div>
  <div class="iron-grp">what gets worse</div>
  <div class="iron-row">
    <div class="iron-name">Time to first</div>
    <div class="iron-bar-outer"><div class="iron-bar-inner" id="iron-ttft-bar"></div></div>
    <div class="iron-val" id="iron-ttft-val"></div>
  </div>
  <div class="iron-row">
    <div class="iron-name">GPU memory</div>
    <div class="iron-bar-outer"><div class="iron-bar-inner" id="iron-vram-bar"></div></div>
    <div class="iron-val" id="iron-vram-val"></div>
  </div>
  <div class="iron-note">TPOT holds at ~5ms regardless of batch. Decode is memory-bound: the GPU reads the full model for every token, no matter how many requests share the read.</div>
  <div class="iron-insight" id="iron-insight"></div>
  <div style="font-size:11px;color:var(--inf-muted);font-family:monospace;margin-top:8px;opacity:0.6">Model: 8B params, FP16, single H100 80GB, 4K context</div>
</div>
<script>
function updateIron(){
  var B=parseInt(document.getElementById('iron-batch-slider').value);
  document.getElementById('iron-batch').textContent=B;
  var tput=B*200;
  var tputPct=Math.min(tput/52000*100,100);
  document.getElementById('iron-tput-bar').style.width=tputPct+'%';
  document.getElementById('iron-tput-bar').style.background='var(--inf-green)';
  document.getElementById('iron-tput-val').textContent=tput.toLocaleString()+' tok/s';
  var cost=4.0/B;
  var costPct=Math.min(cost/4.1*100,100);
  var cc=cost>2?'var(--inf-red)':cost>0.3?'var(--inf-orange)':'var(--inf-green)';
  document.getElementById('iron-cost-bar').style.width=costPct+'%';
  document.getElementById('iron-cost-bar').style.background=cc;
  document.getElementById('iron-cost-val').textContent='$'+(cost<0.1?cost.toFixed(3):cost.toFixed(2))+'/M tokens';
  var ttft=10+2*B;
  var ttftPct=Math.min(ttft/550*100,100);
  var tc=ttft<100?'var(--inf-green)':ttft<300?'var(--inf-orange)':'var(--inf-red)';
  document.getElementById('iron-ttft-bar').style.width=ttftPct+'%';
  document.getElementById('iron-ttft-bar').style.background=tc;
  document.getElementById('iron-ttft-val').textContent=ttft+'ms';
  var vram=21+B*0.5;
  var vramPct=Math.min(vram/80*100,100);
  var vc=vram<55?'var(--inf-green)':vram<72?'var(--inf-orange)':'var(--inf-red)';
  var vl=vram.toFixed(0)+'GB / 80GB';
  if(vram>80) vl=vram.toFixed(0)+'GB / 80GB — OOM';
  document.getElementById('iron-vram-bar').style.width=vramPct+'%';
  document.getElementById('iron-vram-bar').style.background=vc;
  document.getElementById('iron-vram-val').textContent=vl;
  var msg='';
  if(B<=2) msg='The GPU loads 16GB of weights to produce '+B+' token(s), then waits. You are renting a datacenter GPU for one conversation.';
  else if(B<=16) msg='Weight reads amortized across '+B+' requests. Cost already $'+cost.toFixed(2)+'/M. GPU still has headroom.';
  else if(vram<72) msg='High throughput, low cost, VRAM within budget. Production systems aim for this zone.';
  else if(vram<80) msg='Approaching VRAM limit. KV cache is '+(vram-21).toFixed(0)+'GB. A few more requests trigger eviction or OOM.';
  else msg='KV cache exceeds VRAM. You need PagedAttention, KV quantization, or more GPUs. Every technique in this post pushes this wall further out.';
  document.getElementById('iron-insight').textContent=msg;
}
updateIron();
</script>

Batching is free throughput - until memory says stop. The entire inference optimization stack exists to push that memory wall further out, letting you pack more requests per GPU without blowing latency SLOs.

Four metrics track where your system sits in this tradeoff:

- **TTFT** (Time to First Token): time before the user sees output. The prefill phase drives this.
- **TPOT** (Time per Output Token): time between consecutive tokens during decode. Memory bandwidth drives this.
- **Goodput**: requests completed within the SLO deadline per second. Raw throughput means nothing if half the responses miss the latency target.
- **MFU** (Model FLOPS Utilization): fraction of the GPU's theoretical peak FLOPS your workload uses. Most inference hits 30-60%.

Averages lie. One slow request in ten makes an app feel broken. Production systems target P99 TTFT and P99 TPOT: the latency that 99th-percentile requests experience.

---

## two phases: prefill and decode

When you send a prompt to an LLM, inference runs in two phases:

<div class="diag">
<style>
@keyframes inf-pulse-green { 0%,100% { opacity:0.7; } 50% { opacity:1; } }
@keyframes inf-pulse-red { 0%,100% { opacity:0.7; } 50% { opacity:1; } }
.inf-prefill-box { animation: inf-pulse-green 2s ease-in-out infinite; }
.inf-decode-box { animation: inf-pulse-red 2.5s ease-in-out infinite; }
</style>
<svg viewBox="0 0 800 230" style="max-width:780px">
  <defs>
    <style>
      .inf-phase-title { font-family: monospace; font-size: 18px; font-weight: 700; }
      .inf-phase-detail { font-family: monospace; font-size: 12px; fill: var(--inf-muted, #888); }
      .inf-phase-metric { font-family: monospace; font-size: 13px; font-weight: 600; }
    </style>
    <marker id="inf-arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="var(--inf-muted)"/></marker>
  </defs>
  <rect x="20" y="20" width="310" height="190" rx="10" fill="var(--inf-green, #4caf78)" opacity="0.15" stroke="var(--inf-green)" stroke-width="2" class="inf-prefill-box"/>
  <text x="175" y="50" text-anchor="middle" class="inf-phase-title" fill="var(--inf-green)">PREFILL</text>
  <text x="175" y="75" text-anchor="middle" class="inf-phase-detail">Process entire prompt at once</text>
  <rect x="60" y="95" width="80" height="40" rx="4" fill="var(--inf-green)" opacity="0.3"/>
  <text x="100" y="120" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">Matrix</text>
  <text x="155" y="120" text-anchor="middle" font-family="monospace" font-size="16" fill="var(--inf-text)">x</text>
  <rect x="170" y="95" width="80" height="40" rx="4" fill="var(--inf-green)" opacity="0.3"/>
  <text x="210" y="120" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">Matrix</text>
  <text x="175" y="155" text-anchor="middle" class="inf-phase-metric" fill="var(--inf-green)">COMPUTE-BOUND</text>
  <text x="175" y="175" text-anchor="middle" class="inf-phase-detail">GPU cores saturated</text>
  <text x="175" y="195" text-anchor="middle" class="inf-phase-detail">Determines TTFT</text>
  <line x1="345" y1="115" x2="415" y2="115" stroke="var(--inf-muted)" stroke-width="2" marker-end="url(#inf-arrow)"/>
  <text x="380" y="95" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-orange)">KV Cache</text>
  <text x="380" y="140" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-muted)">built here</text>
  <rect x="430" y="20" width="340" height="190" rx="10" fill="var(--inf-red, #f87171)" opacity="0.15" stroke="var(--inf-red)" stroke-width="2" class="inf-decode-box"/>
  <text x="600" y="50" text-anchor="middle" class="inf-phase-title" fill="var(--inf-red)">DECODE</text>
  <text x="600" y="75" text-anchor="middle" class="inf-phase-detail">Generate tokens one at a time</text>
  <rect x="475" y="95" width="24" height="40" rx="3" fill="var(--inf-red)" opacity="0.3"/>
  <text x="487" y="120" text-anchor="middle" font-family="monospace" font-size="10" fill="var(--inf-text)">Vec</text>
  <text x="515" y="120" text-anchor="middle" font-family="monospace" font-size="16" fill="var(--inf-text)">x</text>
  <rect x="530" y="95" width="80" height="40" rx="4" fill="var(--inf-red)" opacity="0.3"/>
  <text x="570" y="120" text-anchor="middle" font-family="monospace" font-size="11" fill="var(--inf-text)">Matrix</text>
  <text x="600" y="155" text-anchor="middle" class="inf-phase-metric" fill="var(--inf-red)">MEMORY-BOUND</text>
  <text x="600" y="175" text-anchor="middle" class="inf-phase-detail">GPU idle, waiting for data</text>
  <text x="600" y="195" text-anchor="middle" class="inf-phase-detail">Determines TPOT</text>
</svg>
<div class="diag-caption">Prefill: matrix-matrix multiply (compute-bound). Decode: vector-matrix multiply (memory-bound).</div>
</div>

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/inference_prefill_decode_poster.jpg">
  <source src="/assets/images/inference/inference_prefill_decode.mp4" type="video/mp4">
</video>
</div>

**Prefill** processes the entire input prompt in one parallel pass. Big matrices multiplied together. The GPU's tensor cores are fully occupied. This is compute-bound and determines TTFT.

**Decode** generates tokens one at a time. Each step runs a tiny vector through the entire model. The GPU loads billions of parameters from memory to perform a handful of operations on each one. Then it loads them again for the next token. This is memory-bandwidth-bound and determines TPOT.

**Output selection**: the final layer produces a logit vector with one value per vocabulary token. Softmax normalizes these into probabilities. Then you pick:
- **Temperature**: divide logits by T before softmax. Lower T = more deterministic. T=0 = greedy.
- **Top-K**: keep only the K most likely tokens, re-normalize.
- **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative probability reaches p.

---

## gpu hardware

**Streaming Multiprocessors (SMs)** are the GPU's building blocks. An H100 has 132 SMs, each containing:

- **CUDA Cores**: general-purpose parallel computation.
- **Tensor Cores**: operate on small matrices (4x4, 8x8, 16x16). Handle the matrix multiply-accumulate operations that dominate inference.

### memory hierarchy

Four levels, each trading capacity for speed:

| Level | H100 Size | H100 Bandwidth | Latency |
|-------|-----------|----------------|---------|
| Registers | ~256KB per SM | ~20 TB/s | 1 cycle |
| L1 / Shared Memory (SRAM) | 256KB per SM, 33MB total | ~19 TB/s | ~30 cycles |
| L2 Cache | 50MB | ~12 TB/s | ~200 cycles |
| HBM (VRAM) | 80GB | 3.35 TB/s | ~400 cycles |

Data flows up this hierarchy. Every computation loads operands from memory, computes, and stores results. The bottleneck in inference is memory bandwidth, not compute. SRAM is 6x faster than HBM, but HBM is 2400x larger. FlashAttention exploits this gap by keeping work in SRAM.

### vram sizing

Total VRAM must hold:
- **Model weights**: parameters x bytes_per_parameter. A 70B model at FP16 = 140GB.
- **KV cache**: grows with batch size and sequence length. Budget 50% of weight memory for headroom.
- **Activations and workspace**: temporary tensors during the forward pass.

**Rule of thumb**: if weights fill more than 60% of VRAM, you will hit OOM under production load from KV cache growth.

<style>
.vram-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.vram-widget label{font-size:13px;display:block;margin-bottom:4px;color:var(--inf-muted);font-family:monospace}
.vram-widget select,.vram-widget input[type=number]{background:var(--inf-border);color:var(--inf-text);border:1px solid var(--inf-muted);border-radius:5px;padding:6px 10px;font-family:monospace;font-size:13px;margin:4px 0 12px}
.vram-widget .result{font-size:15px;font-weight:600;margin-top:12px;font-family:monospace;color:var(--inf-text)}
.vram-widget .bar-outer{height:20px;background:var(--inf-border);border-radius:10px;margin-top:8px;overflow:hidden;position:relative}
.vram-widget .bar-seg{height:100%;float:left;transition:width 0.3s}
.vram-widget .legend{display:flex;gap:16px;margin-top:8px;font-size:12px;font-family:monospace;color:var(--inf-muted)}
.vram-widget .legend span{display:inline-flex;align-items:center;gap:4px}
.vram-widget .legend .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
</style>
<div class="vram-widget">
  <label>Model parameters (billions): <input type="number" id="vram-params" value="70" min="1" max="1000" style="width:80px" oninput="updateVRAM()"></label>
  <label>Precision: <select id="vram-prec" onchange="updateVRAM()"><option value="4">FP32 (4 bytes)</option><option value="2" selected>FP16/BF16 (2 bytes)</option><option value="1">INT8 (1 byte)</option><option value="0.5">INT4 (0.5 bytes)</option></select></label>
  <label>Target GPU: <select id="vram-gpu" onchange="updateVRAM()"><option value="80">H100 (80GB)</option><option value="141">H200 (141GB)</option><option value="192">B200 (192GB)</option><option value="24">RTX 4090 (24GB)</option><option value="192">M3 Ultra (192GB)</option></select></label>
  <div class="bar-outer"><div class="bar-seg" id="vram-w-bar" style="background:var(--inf-blue)"></div><div class="bar-seg" id="vram-kv-bar" style="background:var(--inf-orange)"></div></div>
  <div class="legend">
    <span><span class="dot" style="background:var(--inf-blue)"></span>Weights</span>
    <span><span class="dot" style="background:var(--inf-orange)"></span>KV headroom (50%)</span>
    <span><span class="dot" style="background:var(--inf-border)"></span>Free</span>
  </div>
  <div class="result" id="vram-result"></div>
</div>
<script>
function updateVRAM(){
  var params=parseFloat(document.getElementById('vram-params').value)*1e9;
  var bpp=parseFloat(document.getElementById('vram-prec').value);
  var gpu=parseFloat(document.getElementById('vram-gpu').value);
  var wGB=params*bpp/1e9;
  var kvGB=wGB*0.5;
  var total=wGB+kvGB;
  var gpus=Math.ceil(total/gpu);
  var wPct=Math.min(wGB/gpu*100,100);
  var kvPct=Math.min(kvGB/gpu*100,100-wPct);
  document.getElementById('vram-w-bar').style.width=wPct+'%';
  document.getElementById('vram-kv-bar').style.width=kvPct+'%';
  var r='Weights: '+wGB.toFixed(1)+'GB | KV headroom: '+kvGB.toFixed(1)+'GB | Total: '+total.toFixed(1)+'GB';
  if(gpus>1) r+=' | Need '+gpus+'x GPUs (TP='+gpus+')';
  else r+=' | Fits on 1 GPU with '+(gpu-total).toFixed(0)+'GB free';
  document.getElementById('vram-result').textContent=r;
}
updateVRAM();
</script>

Try changing precision from FP16 to INT4: the 70B model drops from 140GB to 35GB, fitting on a single RTX 4090.

### gpu generations

Two tiers matter for inference:

**Hopper (H100, H200)**: FP8 support via 4th-gen Tensor Cores (2x FLOPS of FP16). Transformer Engine for dynamic FP8/FP16 switching. 900 GB/s NVLink. FlashAttention 3 exploits Hopper-specific features: WGMMA (warp-group matrix multiply-accumulate), TMA (tensor memory accelerator), FP8.

**Blackwell (B200, B300)**: FP4 support (double FP8 throughput). 5th-gen Tensor Cores with micro-tensor scaling. 1800 GB/s NVLink (2x Hopper). The current gold standard for inference.

### multi-gpu interconnect

| Connection | Bandwidth | Use |
|------------|-----------|-----|
| NVLink (Blackwell) | 1800 GB/s | GPU-to-GPU within node |
| NVLink (Hopper) | 900 GB/s | GPU-to-GPU within node |
| InfiniBand NDR | 400 Gb/s | Node-to-node across cluster |
| PCIe Gen5 | 128 GB/s | CPU-GPU, consumer cards |

Tensor parallelism uses NVLink (fast, within a node). Pipeline parallelism uses InfiniBand (slower, across nodes). The bandwidth gap between these two shapes every parallelism decision.

---

## the roofline: why your gpu is bored

**Arithmetic intensity** = FLOPs / bytes moved. If your workload does few operations per byte loaded, you're memory-bound. LLM prefill sits around 200-300 ops/byte (compute-bound). LLM decode sits around 1-10 ops/byte (memory-bound, the GPU is idle most of the time).

### the byte budget for a single decode step

For a 70B model at FP16:

- **Bytes to read**: 70 x 10^9 x 2 = 140GB (the entire model weight, loaded from HBM for one token).
- **FLOPs performed**: 70 x 10^9 x 2 = 140 GFLOPS (2 FLOPs per parameter for a matrix-vector multiply).
- **Arithmetic intensity**: 140 GFLOPS / 140 GB = **1 op/byte**. The H100's ridge point is 295.
- **Time at peak bandwidth**: 140 GB / 3.35 TB/s = **41.8 ms per token** = ~24 tokens/second (batch=1).

That 24 tok/s number comes from memory bandwidth, not compute. The GPU does 140 GFLOPS of the 990 TFLOPS available (0.014% utilization). You could double the compute and TPOT wouldn't change.

The fix for this ratio: **batching**. With batch=32, you still read 140GB of weights (shared), but perform 32x more compute on them. Arithmetic intensity jumps to 32 ops/byte. Still memory-bound, but the GPU utilization rises from 0.014% to 10.8%. Each token costs ~1.3ms. Amortize weight reads across many requests.

<style>
.tpot-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.tpot-widget label{font-size:13px;display:block;margin-bottom:4px;color:var(--inf-muted);font-family:monospace}
.tpot-widget input[type=range]{width:100%;margin:4px 0 12px}
.tpot-widget .result{font-size:15px;font-weight:600;margin-top:8px;font-family:monospace;color:var(--inf-text)}
</style>
<div class="tpot-widget">
  <label>Model size (billions): <strong id="tpot-params">70</strong></label>
  <input type="range" min="1" max="405" value="70" id="tpot-params-slider" oninput="updateTPOT()">
  <label>Precision bytes: <strong id="tpot-bpp">2</strong> (FP16)</label>
  <input type="range" min="0.5" max="4" step="0.5" value="2" id="tpot-bpp-slider" list="tpot-prec-list" oninput="updateTPOT()">
  <datalist id="tpot-prec-list"><option value="0.5"><option value="1"><option value="2"><option value="4"></datalist>
  <label>Batch size: <strong id="tpot-batch">1</strong></label>
  <input type="range" min="1" max="128" value="1" id="tpot-batch-slider" oninput="updateTPOT()">
  <label>GPU: H100 (3.35 TB/s bandwidth, 990 TFLOPS FP16)</label>
  <div class="result" id="tpot-result"></div>
</div>
<script>
function updateTPOT(){
  var p=parseFloat(document.getElementById('tpot-params-slider').value);
  var bpp=parseFloat(document.getElementById('tpot-bpp-slider').value);
  var B=parseInt(document.getElementById('tpot-batch-slider').value);
  document.getElementById('tpot-params').textContent=p;
  document.getElementById('tpot-bpp').textContent=bpp;
  document.getElementById('tpot-batch').textContent=B;
  var precLabel={0.5:'INT4',1:'INT8',2:'FP16',4:'FP32'};
  if(precLabel[bpp]) document.getElementById('tpot-bpp').textContent=bpp+' ('+precLabel[bpp]+')';
  var weightGB=p*1e9*bpp/1e9;
  var bw=3.35e12; // bytes/s
  var tpotMs=weightGB*1e9/bw*1000; // ms per token at batch=1
  var tpotBatch=tpotMs/1; // weight read is shared across batch
  var toksPerSec=B/(tpotMs/1000);
  var ai=B; // approx arithmetic intensity = batch_size (for decode)
  var utilPct=Math.min(ai/295*100,100);
  document.getElementById('tpot-result').textContent=
    'Weight read: '+weightGB.toFixed(1)+'GB | TPOT: '+(tpotMs).toFixed(1)+'ms | '+
    toksPerSec.toFixed(0)+' tok/s total | AI: '+ai.toFixed(0)+' ops/byte ('+utilPct.toFixed(1)+'% of peak)';
}
updateTPOT();
</script>

Slide batch from 1 to 32. Tokens per second scales linearly with batch size until you hit the compute ceiling or run out of KV cache VRAM. Production systems batch aggressively for this reason.

### what batch size do frontier models run at?

The hardware dictates the optimal batch size. The H100's FLOPs-to-bandwidth ratio is ~295. To reach peak utilization, you need arithmetic intensity of 295, which means batch size ~295. In practice, [Reiner Pope (MatX)](https://gist.github.com/dwarkeshsp/79100f0fdeed69d76241903bb0604dbe) estimates production systems run at **2,000-3,000 tokens in-flight** per GPU. This number is roughly independent of model size: a 70B model and a 405B model want similar batch sizes per GPU (the larger model just uses more GPUs to shard). Sparsity (MoE) is the main thing that shifts this number, since only a fraction of parameters are active per token.

At batch=1, you pay the full weight read for one token. At batch=2048, that same weight read produces 2048 tokens. Per-token cost drops by 2048x. API providers price decode tokens 3-5x higher than prefill tokens because decode is the expensive phase, and batch sizes during decode determine whether the business is profitable.

---

## the kv cache problem

### why it exists

Attention lets each token attend to every previous token. Without caching, each decode step would recompute attention for the entire history: O(N^2) per token. The **KV cache** stores keys and values from all previous tokens, reducing attention to O(N) per step. Built during prefill, read and extended during decode.

The KV cache is the single largest consumer of GPU memory during inference. It constrains batch size and context length more than the model weights do.

### how big does it get?

<style>
.kv-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.kv-widget label{font-size:13px;display:block;margin-bottom:4px;color:var(--inf-muted);font-family:monospace}
.kv-widget input[type=range]{width:100%;margin:4px 0 12px}
.kv-widget .result{font-size:15px;font-weight:600;margin-top:8px;font-family:monospace;color:var(--inf-text)}
.kv-widget .bar-outer{height:16px;background:var(--inf-border);border-radius:8px;margin-top:8px;overflow:hidden}
.kv-widget .bar-inner{height:100%;border-radius:8px;transition:width 0.3s,background 0.3s}
</style>
<div class="kv-widget">
  <label>Batch size: <strong id="kv-batch">1</strong></label>
  <input type="range" min="1" max="64" value="1" id="kv-batch-slider" oninput="updateKV()">
  <label>Sequence length: <strong id="kv-seq">4096</strong></label>
  <input type="range" min="512" max="131072" value="4096" step="512" id="kv-seq-slider" oninput="updateKV()">
  <label>Model: Llama 70B (L=80, H_kv=8, d=128, FP16)</label>
  <div class="bar-outer"><div class="bar-inner" id="kv-bar"></div></div>
  <div class="result" id="kv-result"></div>
  <div style="font-size:12px;color:var(--inf-muted);margin-top:4px;font-family:monospace" id="kv-note"></div>
</div>
<script>
function updateKV(){
  var B=parseInt(document.getElementById('kv-batch-slider').value);
  var N=parseInt(document.getElementById('kv-seq-slider').value);
  document.getElementById('kv-batch').textContent=B;
  document.getElementById('kv-seq').textContent=N.toLocaleString();
  var bytes=2*B*80*8*128*N*2;
  var gb=bytes/1e9;
  var weights=140;
  var vram=80;
  var pct=(gb/vram)*100;
  var bar=document.getElementById('kv-bar');
  bar.style.width=Math.min(pct,100)+'%';
  bar.style.background=pct>100?'var(--inf-red)':pct>60?'var(--inf-orange)':'var(--inf-green)';
  document.getElementById('kv-result').textContent='KV cache: '+gb.toFixed(1)+' GB ('+pct.toFixed(0)+'% of 80GB H100 VRAM)';
  var note='';
  if(gb>vram) note='OOM. KV cache alone exceeds VRAM. Model weights (140GB) need separate GPUs.';
  else if(gb+weights>vram) note='Weights (140GB) + KV cache ('+gb.toFixed(1)+'GB) = '+(gb+weights).toFixed(0)+'GB. Needs TP across multiple GPUs.';
  else note='Weights + KV cache = '+(gb+weights).toFixed(0)+'GB. Fits in VRAM with '+(vram-gb-weights).toFixed(0)+'GB headroom.';
  document.getElementById('kv-note').textContent=note;
}
updateKV();
</script>

Drag the batch slider to 32. The KV cache hits **342 GB**: more than double the model weights (140GB). The KV cache, not the model, is the memory bottleneck.

The formula:

$$\text{KV cache} = 2 \times B \times L \times H_{kv} \times d \times N \times \text{bytes}$$

B=batch, L=layers, $$H_{kv}$$=KV heads, d=head dim, N=seq length. For Llama 70B (L=80, $$H_{kv}$$=8, d=128) at FP16, batch=1, N=4096: 2 x 1 x 80 x 8 x 128 x 4096 x 2 = **10.7 GB**.

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/ai_kv_cache_poster.jpg">
  <source src="/assets/images/inference/ai_kv_cache.mp4" type="video/mp4">
</video>
</div>

### pagedattention: os for gpu memory

Standard allocation pre-reserves the maximum possible cache per request. If your model supports 128K context, every request reserves GB of VRAM upfront, most of it wasted on short conversations.

**PagedAttention** (Kwon et al., 2023, the paper behind vLLM) borrows from OS virtual memory:

<div class="diag">
<svg viewBox="0 0 750 270" style="max-width:730px">
  <defs>
    <style>
      .pa-label { font-family: monospace; font-size: 13px; font-weight: 600; fill: var(--inf-text, #e8e8e8); }
      .pa-sub { font-family: monospace; font-size: 11px; fill: var(--inf-muted, #888); }
      .pa-block { rx: 4; stroke-width: 1.5; }
    </style>
  </defs>
  <text x="130" y="20" text-anchor="middle" class="pa-label">Block Table (logical)</text>
  <rect x="20" y="35" width="220" height="38" rx="4" fill="none" stroke="var(--inf-blue)" stroke-width="1.5"/>
  <text x="35" y="58" class="pa-sub" fill="var(--inf-blue)">Req 1: [B0, B3, B7, B9]</text>
  <rect x="20" y="82" width="220" height="38" rx="4" fill="none" stroke="var(--inf-purple)" stroke-width="1.5"/>
  <text x="35" y="105" class="pa-sub" fill="var(--inf-purple)">Req 2: [B0, B3, B5, B8]</text>
  <text x="130" y="142" text-anchor="middle" class="pa-sub">B0, B3 shared (copy-on-write)</text>
  <line x1="250" y1="72" x2="325" y2="72" stroke="var(--inf-muted)" stroke-width="1.5" marker-end="url(#inf-arrow)"/>
  <text x="530" y="20" text-anchor="middle" class="pa-label">GPU VRAM (physical)</text>
  <rect x="340" y="38" width="58" height="38" class="pa-block" fill="var(--inf-green)" opacity="0.6" stroke="var(--inf-green)"/>
  <text x="369" y="62" text-anchor="middle" class="pa-sub">B0</text>
  <rect x="408" y="38" width="58" height="38" class="pa-block" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/>
  <text x="437" y="62" text-anchor="middle" class="pa-sub">B1</text>
  <rect x="476" y="38" width="58" height="38" class="pa-block" fill="var(--inf-border)" opacity="0.3" stroke="var(--inf-border)"/>
  <text x="505" y="62" text-anchor="middle" class="pa-sub">free</text>
  <rect x="544" y="38" width="58" height="38" class="pa-block" fill="var(--inf-green)" opacity="0.6" stroke="var(--inf-green)"/>
  <text x="573" y="62" text-anchor="middle" class="pa-sub">B3</text>
  <rect x="612" y="38" width="58" height="38" class="pa-block" fill="var(--inf-border)" opacity="0.3" stroke="var(--inf-border)"/>
  <text x="641" y="62" text-anchor="middle" class="pa-sub">free</text>
  <rect x="680" y="38" width="58" height="38" class="pa-block" fill="var(--inf-purple)" opacity="0.3" stroke="var(--inf-purple)"/>
  <text x="709" y="62" text-anchor="middle" class="pa-sub">B5</text>
  <rect x="340" y="88" width="58" height="38" class="pa-block" fill="var(--inf-border)" opacity="0.3" stroke="var(--inf-border)"/>
  <text x="369" y="112" text-anchor="middle" class="pa-sub">free</text>
  <rect x="408" y="88" width="58" height="38" class="pa-block" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/>
  <text x="437" y="112" text-anchor="middle" class="pa-sub">B7</text>
  <rect x="476" y="88" width="58" height="38" class="pa-block" fill="var(--inf-purple)" opacity="0.3" stroke="var(--inf-purple)"/>
  <text x="505" y="112" text-anchor="middle" class="pa-sub">B8</text>
  <rect x="544" y="88" width="58" height="38" class="pa-block" fill="var(--inf-blue)" opacity="0.3" stroke="var(--inf-blue)"/>
  <text x="573" y="112" text-anchor="middle" class="pa-sub">B9</text>
  <text x="530" y="150" text-anchor="middle" class="pa-sub">Green = shared prefix blocks</text>
  <text x="530" y="168" text-anchor="middle" class="pa-sub">Blocks scattered, not contiguous</text>
</svg>
<div class="diag-caption">PagedAttention: a block table maps logical positions to scattered physical blocks. Shared prefixes use copy-on-write.</div>
</div>

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/inference_paged_attention_poster.jpg">
  <source src="/assets/images/inference/inference_paged_attention.mp4" type="video/mp4">
</video>
</div>

- Allocate blocks on demand as the sequence grows. No upfront reservation.
- Near-zero memory waste (fragmentation < 1 block per request).
- **Copy-on-write**: two requests sharing a system prompt? They share the same KV blocks. Fork only when they diverge.

vLLM's throughput gains come mainly from this: more requests fit in memory because there's less waste.

### prefix caching

Many requests share a common prefix: system prompts, few-shot examples, shared context. Without prefix caching, every request recomputes the KV cache for this shared prefix.

**Automatic prefix caching** stores KV blocks keyed by their token content. New requests with a matching prefix reuse the cached blocks. The request starts decode from where the prefix ends, skipping prefill for the shared portion.

**RadixAttention** (SGLang) organizes cached prefixes in a radix tree. Each node represents a token sequence, branching points correspond to where requests diverge. You get prefix sharing across requests with partial overlap, not only exact matches.

### kv cache tiers

Not all KV cache entries need the fastest memory. Same idea as a CDN edge / origin / cold storage hierarchy, but for GPU memory:

| Tier | Storage | Bandwidth | Use |
|------|---------|-----------|-----|
| G1 | GPU HBM | ~3 TB/s | Active sequences |
| G2 | Peer GPU (NVLink) | ~900 GB/s | Overflow to neighboring GPU |
| G3 | CPU DRAM | ~200 GB/s | Evicted but recent sequences |
| G4 | Disk / Network | ~10 GB/s | Cold storage |

When a sequence gets evicted from G1, demote its KV blocks to G3 (CPU memory). If the user returns (common in chat), promote the blocks back instead of recomputing prefill.

### streamingllm

For infinite-length streaming tasks, the KV cache grows without bound. StreamingLLM (Xiao et al.) discovered that the first few tokens in any sequence act as **attention sinks**: they absorb disproportionate attention probability regardless of their content.

The fix: keep the attention sink tokens (first 4-8 tokens) plus a sliding window of recent tokens. Drop everything in between. Fixed memory, infinite-length generation.

### kv cache quantization

The KV cache contains activations, not static weights. Quantizing it to INT8 or FP8 halves the memory footprint with minimal quality loss (typically <0.5% perplexity increase). This doubles the effective batch size or context length for the same VRAM budget.

---

## flashattention: the tiling trick

Standard attention computes $$S = QK^T$$, then $$P = \text{softmax}(S)$$, then $$O = PV$$. That intermediate $$S$$ matrix is $$N \times N$$ where N is the sequence length. At N=4096, that's 64MB at FP16. At N=128K, it's 32GB, bigger than most GPUs.

The standard algorithm writes this full matrix to HBM (slow memory), reads it back, applies softmax, writes again, reads again, multiplies by V. Four HBM round-trips for a matrix that only exists to be immediately consumed.

<div class="diag">
<svg viewBox="0 0 760 250" style="max-width:740px">
  <defs>
    <style>
      .fa-label { font-family: monospace; font-size: 14px; font-weight: 600; }
      .fa-sub { font-family: monospace; font-size: 11px; fill: var(--inf-muted, #888); }
    </style>
  </defs>
  <text x="170" y="20" text-anchor="middle" class="fa-label" fill="var(--inf-red)">Standard Attention</text>
  <rect x="70" y="35" width="200" height="190" rx="6" fill="var(--inf-red)" opacity="0.12" stroke="var(--inf-red)" stroke-width="1.5"/>
  <line x1="70" y1="82" x2="270" y2="82" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="70" y1="130" x2="270" y2="130" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="70" y1="177" x2="270" y2="177" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="120" y1="35" x2="120" y2="225" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="170" y1="35" x2="170" y2="225" stroke="var(--inf-red)" opacity="0.2"/>
  <line x1="220" y1="35" x2="220" y2="225" stroke="var(--inf-red)" opacity="0.2"/>
  <text x="170" y="125" text-anchor="middle" class="fa-sub" fill="var(--inf-red)">Full N x N matrix</text>
  <text x="170" y="143" text-anchor="middle" class="fa-sub" fill="var(--inf-red)">materialized in HBM</text>
  <text x="380" y="130" text-anchor="middle" font-family="monospace" font-size="28" fill="var(--inf-muted)">vs</text>
  <text x="580" y="20" text-anchor="middle" class="fa-label" fill="var(--inf-green)">FlashAttention</text>
  <rect x="480" y="35" width="200" height="190" rx="6" fill="none" stroke="var(--inf-border)" stroke-width="1" stroke-dasharray="4,3"/>
  <rect x="490" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="540" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="590" y="45" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="640" y="45" width="30" height="38" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="490" y="93" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="540" y="93" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="590" y="93" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="640" y="93" width="30" height="38" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="490" y="141" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="540" y="141" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="590" y="141" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <rect x="640" y="141" width="30" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="490" y="189" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.1"/>
  <rect x="540" y="189" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.15"/>
  <rect x="590" y="189" width="40" height="38" rx="3" fill="var(--inf-green)" opacity="0.3"/>
  <rect x="640" y="189" width="30" height="38" rx="3" fill="var(--inf-green)" opacity="0.7"/>
  <text x="580" y="125" text-anchor="middle" class="fa-sub" fill="var(--inf-green)">Tiled blocks</text>
  <text x="580" y="143" text-anchor="middle" class="fa-sub" fill="var(--inf-green)">computed in SRAM</text>
  <text x="580" y="161" text-anchor="middle" class="fa-sub" fill="var(--inf-green)">never full matrix</text>
</svg>
<div class="diag-caption">Left: standard attention materializes NxN in HBM. Right: FlashAttention tiles it in SRAM, block by block.</div>
</div>

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/ai_flash_attention_poster.jpg">
  <source src="/assets/images/inference/ai_flash_attention.mp4" type="video/mp4">
</video>
</div>

FlashAttention (Dao et al., 2022) never builds the full matrix. Instead:

1. **Tile** Q, K, V into small blocks that fit in SRAM (~256KB per SM on H100).
2. For each Q block, iterate over K, V blocks. Compute attention scores **in SRAM**, never writing the intermediate matrix to HBM.
3. **Online softmax** makes this possible. Standard softmax needs the max over the entire row before normalizing. The online softmax algorithm maintains running statistics (max and sum) that update as each block arrives. The final result is exact.

Memory drops from O(N^2) to O(N). Same exact output, fewer memory trips. FlashAttention is now the default in every major inference engine.

### reducing kv heads: mqa, gqa, mla

Model designers reduce attention cost at the architecture level:

- **Multi-Query Attention (MQA)**: all heads share one K, V pair. Cuts KV cache by the number of heads. PaLM uses MQA.
- **Grouped-Query Attention (GQA)**: heads share K, V in groups (e.g., 8 heads, 2 KV groups). Llama 2/3 use GQA.
- **Multi-Latent Attention (MLA)**: projects K, V into a low-dimensional latent space, reconstructs at attention time. Massive KV cache compression. DeepSeek-V2/V3 use MLA.
- **Sliding Window**: each token attends only to the previous w tokens. Reduces O(N^2) to O(Nw). Mistral uses this.

---

## kernel fusion: why it matters

Each GPU kernel reads inputs from HBM, computes, and writes results back to HBM. Three separate kernels (LayerNorm, Linear, GeLU) mean three round-trips through HBM: 6 memory operations total.

A **fused kernel** combines all three operations into one kernel that reads once, computes all three in SRAM, and writes once. Memory traffic drops by ~3x. Same math, fewer bus rides. Database engineers will recognize this: a query planner fusing a scan + filter + projection into one pass over the data instead of materializing intermediate results. Same principle, different hardware.

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/inference_kernel_fusion_poster.jpg">
  <source src="/assets/images/inference/inference_kernel_fusion.mp4" type="video/mp4">
</video>
</div>

**torch.compile** (PyTorch 2.0+) automates fusion: it traces the model, builds a computation graph, and generates optimized kernels. For standard architectures, torch.compile captures much of the benefit of hand-written fused kernels with zero manual work.

**Triton** (OpenAI) lets you write GPU kernels in Python instead of CUDA C++. The compiler handles thread management, memory coalescing, and tiling. Performance sits between naive CUDA and expert hand-tuned CUDA. Many FlashAttention variants ship as Triton code.

---

## quantization: the precision-performance tradeoff

A 70B model at FP32 is 280GB. At FP16, 140GB. At INT4, 35GB. Fewer bits = less memory = less bandwidth consumed = faster decode.

### number formats

| Format | Bits | Range | Use |
|--------|------|-------|-----|
| FP32 | 32 | +/-3.4e38 | Training default |
| FP16 | 16 | +/-65504 | Inference, fine-tuning |
| BF16 | 16 | +/-3.4e38 | Inference default (full range, less precision) |
| FP8 (E4M3) | 8 | +/-448 | Hopper+ compute |
| FP4 | 4 | +/-6 | Blackwell+ |
| INT8 | 8 | -128 to 127 | Weight/activation quantization |
| INT4 | 4 | -8 to 7 | Weight quantization |

BF16 keeps FP32's 8-bit exponent, trading mantissa bits for range. LLM weights span several orders of magnitude, so range matters more than decimal precision.

### what to use

Post-Training Quantization: quantize a trained model without retraining. Three options, in order of "try this first":

| Method | Bits | Quality loss | Calibration | When to use |
|--------|------|-------------|-------------|-------------|
| **FP8 dynamic** | 8 | <1% ppl | None (runtime) | Hopper+ with VRAM to spare. Zero effort, near-lossless. |
| **AWQ** | 4 | ~5% ppl | Minutes | **Default for 4-bit.** Fits large models on consumer GPUs. Used in the code examples below. |
| **GPTQ** | 4-3 | ~5% ppl | Hours | When AWQ weights aren't available, or you need 3-bit. |

AWQ and GPTQ produce similar quality at 4-bit. AWQ calibrates faster and has slightly better ecosystem support in vLLM/SGLang. Most quantized models on HuggingFace ship in one or both formats. QuIP# pushes to 2-bit using random orthogonal transforms - research frontier, not yet production-standard.

**W4A16 vs W8A8**: W4A16 (4-bit weights, FP16 activations) saves memory and speeds up decode. W8A8 (INT8 weights + activations) reduces latency for both prefill and decode but needs hardware support. If you're memory-constrained, W4A16. If you're latency-constrained with VRAM headroom, W8A8.

### accuracy vs speed

<style>
.quant-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.quant-widget .quant-row{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--inf-border);font-family:monospace;font-size:13px;color:var(--inf-text)}
.quant-widget .quant-row:last-child{border-bottom:none}
.quant-widget .quant-bits{width:50px;text-align:right;font-weight:700}
.quant-widget .quant-bar-outer{flex:1;height:14px;background:var(--inf-border);border-radius:7px;overflow:hidden}
.quant-widget .quant-bar-inner{height:100%;border-radius:7px}
.quant-widget .quant-label{width:120px;font-size:12px;color:var(--inf-muted)}
</style>
<div class="quant-widget">
  <div style="font-family:monospace;font-size:14px;font-weight:600;color:var(--inf-text);margin-bottom:8px">Qwen 3.5 9B - Quality vs Speed (representative benchmarks)</div>
  <div class="quant-row">
    <div class="quant-bits">BF16</div>
    <div class="quant-bar-outer"><div class="quant-bar-inner" style="width:100%;background:var(--inf-green)"></div></div>
    <div class="quant-label">8.186 ppl | 107 tok/s</div>
  </div>
  <div class="quant-row">
    <div class="quant-bits">INT8</div>
    <div class="quant-bar-outer"><div class="quant-bar-inner" style="width:99.9%;background:var(--inf-green)"></div></div>
    <div class="quant-label">8.193 ppl | +0.1%</div>
  </div>
  <div class="quant-row">
    <div class="quant-bits">INT4</div>
    <div class="quant-bar-outer"><div class="quant-bar-inner" style="width:95.4%;background:var(--inf-orange)"></div></div>
    <div class="quant-label">8.563 ppl | 177 tok/s</div>
  </div>
  <div class="quant-row">
    <div class="quant-bits" style="color:var(--inf-red)">INT2</div>
    <div class="quant-bar-outer"><div class="quant-bar-inner" style="width:12.4%;background:var(--inf-red)"></div></div>
    <div class="quant-label" style="color:var(--inf-red)">66.1 ppl | UNUSABLE</div>
  </div>
</div>

8-bit: lossless for all practical purposes. 4-bit: ~2x faster, ~5% quality drop. Below 4-bit: the model falls off a cliff.

---

## speculative decoding: guessing ahead

Use a small model to guess ahead, verify all guesses in one big-model pass. CPU branch prediction, but for language: predict the likely next tokens, execute speculatively, flush and retry on misprediction.

<div class="diag">
<style>
@keyframes inf-draft { 0% { opacity:0; } 20% { opacity:1; } 100% { opacity:1; } }
@keyframes inf-verify { 0%,40% { opacity:0; } 60% { opacity:1; } 100% { opacity:1; } }
@keyframes inf-accept { 0%,60% { opacity:0; } 80% { opacity:1; } 100% { opacity:1; } }
.inf-draft-tok { animation: inf-draft 4s ease-in-out infinite; }
.inf-verify-bar { animation: inf-verify 4s ease-in-out infinite; }
.inf-accept-mark { animation: inf-accept 4s ease-in-out infinite; }
</style>
<svg viewBox="0 0 720 220" style="max-width:700px">
  <defs>
    <style>
      .sd-label { font-family: monospace; font-size: 12px; fill: var(--inf-muted, #888); }
      .sd-tok { font-family: monospace; font-size: 14px; font-weight: 600; }
    </style>
  </defs>
  <text x="20" y="24" class="sd-label">Draft model (fast, small):</text>
  <rect x="20" y="34" width="62" height="38" rx="5" fill="var(--inf-green)" opacity="0.7" class="inf-draft-tok" style="animation-delay:0s"/>
  <text x="51" y="58" text-anchor="middle" class="sd-tok" fill="#111">"the"</text>
  <rect x="92" y="34" width="62" height="38" rx="5" fill="var(--inf-green)" opacity="0.7" class="inf-draft-tok" style="animation-delay:0.3s"/>
  <text x="123" y="58" text-anchor="middle" class="sd-tok" fill="#111">"cat"</text>
  <rect x="164" y="34" width="62" height="38" rx="5" fill="var(--inf-green)" opacity="0.7" class="inf-draft-tok" style="animation-delay:0.6s"/>
  <text x="195" y="58" text-anchor="middle" class="sd-tok" fill="#111">"sat"</text>
  <rect x="236" y="34" width="62" height="38" rx="5" fill="var(--inf-green)" opacity="0.7" class="inf-draft-tok" style="animation-delay:0.9s"/>
  <text x="267" y="58" text-anchor="middle" class="sd-tok" fill="#111">"on"</text>
  <rect x="308" y="34" width="62" height="38" rx="5" fill="var(--inf-green)" opacity="0.7" class="inf-draft-tok" style="animation-delay:1.2s"/>
  <text x="339" y="58" text-anchor="middle" class="sd-tok" fill="#111">"a"</text>
  <text x="20" y="104" class="sd-label">Target model (one forward pass verifies all 5):</text>
  <rect x="20" y="114" width="350" height="38" rx="5" fill="var(--inf-blue)" opacity="0.3" class="inf-verify-bar"/>
  <text x="195" y="138" text-anchor="middle" class="sd-tok" fill="var(--inf-blue)">single forward pass</text>
  <text x="420" y="48" class="sd-label" fill="var(--inf-green)" style="font-size:15px">the - accepted</text>
  <text x="420" y="72" class="sd-label" fill="var(--inf-green)" style="font-size:15px">cat - accepted</text>
  <text x="420" y="96" class="sd-label" fill="var(--inf-green)" style="font-size:15px">sat - accepted</text>
  <text x="420" y="120" class="sd-label" fill="var(--inf-red)" style="font-size:15px">on - REJECTED</text>
  <text x="420" y="144" class="sd-label" fill="var(--inf-orange)" style="font-size:15px">+ bonus from target</text>
  <text x="420" y="185" class="sd-label" style="font-size:13px">4 tokens for ~1 big-model pass</text>
</svg>
<div class="diag-caption">Small model drafts fast. Big model verifies in one pass. Rejection sampling guarantees identical output distribution.</div>
</div>

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/ai_speculative_decoding_poster.jpg">
  <source src="/assets/images/inference/ai_speculative_decoding.mp4" type="video/mp4">
</video>
</div>

The acceptance criterion guarantees **identical output** to the target model alone - not an approximation. At 80% acceptance rate, each verification step yields ~5 tokens. For predictable text ("The Eiffel Tower is located in..."), the small model nails it. For creative generation with high temperature, acceptance drops.

<style>
.spec-widget{background:var(--inf-bg);border:1px solid var(--inf-border);border-radius:10px;padding:20px;margin:1.2rem 0}
.spec-widget label{font-size:13px;display:block;margin-bottom:4px;color:var(--inf-muted);font-family:monospace}
.spec-widget input[type=range]{width:100%;margin:4px 0 12px}
.spec-widget .result{font-size:15px;font-weight:600;margin-top:12px;font-family:monospace;color:var(--inf-text)}
.spec-widget .speedbar{height:24px;background:var(--inf-border);border-radius:12px;margin-top:8px;overflow:hidden}
.spec-widget .speedfill{height:100%;border-radius:12px;transition:width 0.3s;display:flex;align-items:center;justify-content:center;font-size:12px;font-family:monospace;color:#111;font-weight:700}
</style>
<div class="spec-widget">
  <label>Draft model acceptance rate: <strong id="spec-alpha">80</strong>%</label>
  <input type="range" min="10" max="99" value="80" id="spec-alpha-slider" oninput="updateSpec()">
  <label>Draft tokens per step (K): <strong id="spec-k">5</strong></label>
  <input type="range" min="1" max="10" value="5" id="spec-k-slider" oninput="updateSpec()">
  <label>Draft model overhead (relative to 1 target pass): <strong id="spec-overhead">10</strong>%</label>
  <input type="range" min="1" max="40" value="10" id="spec-overhead-slider" oninput="updateSpec()">
  <div class="speedbar"><div class="speedfill" id="spec-bar"></div></div>
  <div class="result" id="spec-result"></div>
</div>
<script>
function updateSpec(){
  var a=parseInt(document.getElementById('spec-alpha-slider').value)/100;
  var k=parseInt(document.getElementById('spec-k-slider').value);
  var oh=parseInt(document.getElementById('spec-overhead-slider').value)/100;
  document.getElementById('spec-alpha').textContent=Math.round(a*100);
  document.getElementById('spec-k').textContent=k;
  document.getElementById('spec-overhead').textContent=Math.round(oh*100);
  // Expected accepted tokens = sum_{i=0}^{k-1} a^i * a + correction
  // Simplified: expected = (1 - a^(k+1)) / (1 - a) when alpha < 1
  var expected=(1-Math.pow(a,k+1))/(1-a);
  var costPerStep=1+oh; // 1 target pass + draft overhead
  var tokensPerCost=expected/costPerStep;
  var speedup=tokensPerCost;
  var pct=Math.min(speedup/6*100,100);
  var bar=document.getElementById('spec-bar');
  bar.style.width=pct+'%';
  bar.style.background=speedup>2.5?'var(--inf-green)':speedup>1.5?'var(--inf-orange)':'var(--inf-red)';
  bar.textContent=speedup.toFixed(1)+'x';
  var r='Expected tokens/step: '+expected.toFixed(1)+' | Effective speedup: '+speedup.toFixed(1)+'x';
  if(speedup<1.2) r+=' | Not worth it at this acceptance rate.';
  document.getElementById('spec-result').textContent=r;
}
updateSpec();
</script>

Drag the acceptance rate down to 40%. The speedup collapses.

### draft model alternatives

- **Medusa**: extra prediction heads on the target model. Each head predicts a future token position. No separate draft model.
- **EAGLE**: uses the target model's hidden states (not just tokens) as input to an autoregressive draft head. Better acceptance rates because the drafter sees the target model's internal representation.
- **N-gram speculation**: look up likely continuations from the existing context using N-gram matching. Zero model overhead.
- **Token tree verification**: draft a tree of possibilities instead of a single sequence. Verify the entire tree in one forward pass using a tree attention mask.

### when speculative decoding hurts

- **High batch sizes**: the GPU is already saturated.
- **Creative generation** (high temperature): acceptance rates plummet.
- **Very short outputs**: draft overhead exceeds the savings.
- **Domain mismatch**: no good small model for the domain.

---

## batching

### static batching

Collect N requests, pad all sequences to the same length, process as a single batch. Short sequences sit idle while the longest finishes.

### continuous batching

At every decode step, check if any request is done. When one finishes, slot in a new request from the queue. The batch stays full. The GPU stays busy. 2-5x throughput improvement over static batching. If you've built web servers: continuous batching is to static batching what an event loop (Node.js, asyncio) is to a thread-per-request model. Don't wait for the slowest request to finish before admitting new work.

<div class="diag">
<svg viewBox="0 0 720 310" style="max-width:700px">
  <defs>
    <style>
      .bat-label { font-family: monospace; font-size: 13px; font-weight: 600; fill: var(--inf-text, #e8e8e8); }
      .bat-sub { font-family: monospace; font-size: 11px; fill: var(--inf-muted, #888); }
    </style>
  </defs>
  <text x="20" y="18" class="bat-label" fill="var(--inf-red)">Static Batching</text>
  <rect x="20" y="28" width="280" height="20" rx="3" fill="var(--inf-blue)" opacity="0.6"/>
  <rect x="300" y="28" width="60" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <rect x="20" y="54" width="180" height="20" rx="3" fill="var(--inf-green)" opacity="0.6"/>
  <rect x="200" y="54" width="160" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <rect x="20" y="80" width="350" height="20" rx="3" fill="var(--inf-purple)" opacity="0.6"/>
  <rect x="20" y="106" width="120" height="20" rx="3" fill="var(--inf-orange)" opacity="0.6"/>
  <rect x="140" y="106" width="220" height="20" rx="3" fill="var(--inf-border)" opacity="0.4"/>
  <text x="410" y="65" class="bat-sub">gray = GPU idle (padding)</text>
  <text x="410" y="82" class="bat-sub">all wait for longest request</text>
  <text x="20" y="165" class="bat-label" fill="var(--inf-green)">Continuous Batching</text>
  <rect x="20" y="175" width="280" height="20" rx="3" fill="var(--inf-blue)" opacity="0.6"/>
  <rect x="20" y="201" width="180" height="20" rx="3" fill="var(--inf-green)" opacity="0.6"/>
  <rect x="200" y="201" width="200" height="20" rx="3" fill="var(--inf-yellow)" opacity="0.6"/>
  <rect x="20" y="227" width="350" height="20" rx="3" fill="var(--inf-purple)" opacity="0.6"/>
  <rect x="20" y="253" width="120" height="20" rx="3" fill="var(--inf-orange)" opacity="0.6"/>
  <rect x="140" y="253" width="130" height="20" rx="3" fill="var(--inf-teal)" opacity="0.6"/>
  <rect x="270" y="253" width="100" height="20" rx="3" fill="var(--inf-red)" opacity="0.4"/>
  <rect x="300" y="175" width="100" height="20" rx="3" fill="var(--inf-teal)" opacity="0.6"/>
  <text x="440" y="210" class="bat-sub">new requests slot in immediately</text>
  <text x="440" y="228" class="bat-sub">when old ones finish</text>
  <text x="440" y="246" class="bat-sub">2-5x throughput improvement</text>
</svg>
<div class="diag-caption">Static batching wastes GPU cycles on padding. Continuous batching keeps the batch full.</div>
</div>

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/inference_batching_poster.jpg">
  <source src="/assets/images/inference/inference_batching.mp4" type="video/mp4">
</video>
</div>

### chunked prefill

But there's a subtler problem: **prefill piracy**. A long-prompt request enters the batch. Its prefill takes 500ms of compute. During that time, every decode-phase request in the batch stalls.

**Chunked prefill** breaks long prefills into 512-token chunks, interleaving decode steps between them. No single request monopolizes the GPU. The long prefill takes slightly longer overall, but latency spikes for everyone else disappear. Same principle as preemptive scheduling in an OS kernel: no single process should starve others.

---

## disaggregated serving

Running both phases on the same GPU serves neither well. Split them:

- **Prefill pool**: GPUs optimized for compute (high FLOPS).
- **Decode pool**: GPUs optimized for bandwidth (large VRAM, high HBM bandwidth).

After prefill, the KV cache migrates to a decode GPU via NVLink or RDMA. Each pool runs on hardware tuned for its phase. Each scales independently based on traffic patterns. NVIDIA Dynamo orchestrates this split.


---

## parallelism: when one gpu isn't enough

A 70B model at FP16 is 140GB. A single H100 has 80GB. The model doesn't fit.

Three parallelism strategies split work across GPUs:

<div class="inf-video">
<video autoplay loop muted playsinline preload="none" poster="/assets/images/inference/inference_parallelism_poster.jpg">
  <source src="/assets/images/inference/inference_parallelism.mp4" type="video/mp4">
</video>
</div>

### tensor parallelism (TP)

Split individual weight matrices across GPUs. For a weight matrix W of shape [M, N] across P GPUs:

- **Column parallelism**: split W into P column slices. Each GPU computes a slice of the output. Concatenate results.
- **Row parallelism**: split W into P row slices. Each GPU computes a partial sum. An **AllReduce** combines them.

Every layer requires an AllReduce across all P GPUs. This needs high-bandwidth interconnect (NVLink, not InfiniBand). TP stays within a single node.

### pipeline parallelism (PP)

Split the model by layers. GPU 0 runs layers 1-20, GPU 1 runs layers 21-40, and so on.

Each GPU processes its layer chunk and passes activations to the next. Communication is point-to-point (not AllReduce) and works over InfiniBand.

The problem: **pipeline bubbles**. GPU 1 sits idle until GPU 0 finishes its chunk. GPU 2 waits for GPU 1. With P pipeline stages, utilization drops to ~1/P.

**Micro-batching** reduces bubbles: split each batch into M micro-batches. While GPU 1 processes micro-batch 1, GPU 0 starts micro-batch 2.

$$\text{Bubble fraction} = \frac{P - 1}{P - 1 + M}$$

With M >> P, bubbles become negligible.

### expert parallelism (EP)

For MoE models, distribute experts across GPUs. Each GPU hosts a subset of experts for each MoE layer.

When a token needs expert 7 on GPU 2, the system sends that token's hidden state via **All-to-All** communication, computes the expert output, and returns the result. Communication-intensive but enables high-throughput MoE inference.

### comparison

| Strategy | Communication | Latency Impact | Best For |
|----------|--------------|----------------|----------|
| TP | AllReduce per layer | Moderate (NVLink) | Doesn't fit on 1 GPU, within-node |
| PP | Point-to-point per chunk | High (bubbles) | Very large models, across nodes |
| EP | All-to-All per MoE layer | Variable | MoE models, many GPUs |

**Practical combo**: most large-scale deployments use TP=8 within a node (NVLink) and PP=N across nodes (InfiniBand). MoE models add EP on top.

**The rack interconnect bottleneck.** Hopper topped out at 8 GPUs per NVLink domain. Cross-node communication fell to InfiniBand, 10-20x slower. Blackwell's 72-GPU NVLink domains provide ~8x more scale-up bandwidth, enabling the next generation of model sizes. EP faces the same constraint: All-to-All degrades across racks, so deployments try to keep all experts within one NVLink domain.

---

## mixture of experts

MoE replaces dense feedforward layers with many smaller expert networks plus a learned router. Each token activates only a few experts (e.g., 8 of 128), so per-token compute stays low even with massive total parameter counts. DeepSeek-V3: 671B total parameters, 37B active per token.

The inference tradeoff: you need the full model in VRAM even though each token only uses ~5% of it. In batched inference, different requests activate different experts, so most parameters stay active. MoE models use expert parallelism (All-to-All communication) instead of tensor parallelism (AllReduce).

---

## the optimization priority

If you're deploying an LLM and need to make it faster/cheaper, optimize in this order:

<div class="inf-card">
<strong>0. Profile first.</strong> Your bottleneck might be tokenization, data loading, or network, not the model. <code>torch.profiler</code> and NSight Systems tell you where time goes.<br><br>
<strong>1. Quantize.</strong> W4A16 (4-bit weights, FP16 activations) gives ~2x throughput for ~5% quality loss.<br><br>
<strong>2. Enable continuous batching + chunked prefill.</strong> Use vLLM or SGLang. Table stakes.<br><br>
<strong>3. Enable KV cache prefix sharing.</strong> If multiple requests share a system prompt, prefix caching skips redundant prefill.<br><br>
<strong>4. Right-size your GPU count.</strong> Use the minimum TP degree that fits the model. TP=2 is 2x faster than TP=4 for communication-bound workloads.<br><br>
<strong>5. Consider speculative decoding.</strong> If your workload has predictable outputs (code completion, factual QA), spec decode gives 2-3x speedup.<br><br>
<strong>6. Disaggregate prefill and decode.</strong> Worth it at scale (100+ GPUs). Overkill for small deployments.
</div>

---

## production: the unglamorous parts

Running inference in a notebook and serving it to users are different problems.

### multi-layer caching

Three cache layers, each trading latency for coverage:

| Layer | Storage | Latency | Use |
|-------|---------|---------|-----|
| L1 | Redis | <10ms | Hot predictions, repeated queries (1-day TTL) |
| L2 | KV prefix cache (GPU) | 0ms | Shared system prompts (skips prefill) |
| L3 | Vector DB (Qdrant) | ~50ms | Semantic dedup (>0.95 cosine similarity) |

L3 semantic caching: compute the embedding of an incoming query, search for near-duplicates in the vector DB. If similarity exceeds 0.95, return the cached response. For expensive multi-step agent workloads, this saves 10-15 seconds per cached request.

### cold starts

GPU scheduling (30s-5min) + image pull (30s-2min) + weight loading (1-5min) + engine warmup (10-30s) = 2-8 minutes before a new replica serves traffic. Scale-to-zero doesn't work for latency-sensitive inference. Keep minimum replicas warm.

### ci/cd for models

Tag every Docker image with its Git SHA. Canary 5% of traffic to new versions. Monitor TTFT, TPOT, error rate. If any metric degrades beyond threshold: automatic rollback. You can trace any running container back to its exact source code.

Pin exact dependency versions: `vllm==0.6.3`, not `vllm>=0.6`. A floating pin will break your deployment at 3am when a new release ships with a regression.

---

### failure diagnostics

| Symptom | Likely Cause |
|---------|-------------|
| High TTFT | Queue depth, prefill piracy |
| TPOT regression | KV cache eviction thrashing |
| OOM crashes | KV cache overflow |
| Throughput cliff | TP AllReduce bottleneck |
| Quality degradation | Data drift, weight corruption |

### cost estimation

At 1M requests/day with avg 500 input + 200 output tokens:

**API pricing** ($3/M in, $15/M out) = ~$4,500/day.

**Dedicated** (4x H100 at $3/hr) = $288/day.

15x cheaper in hardware alone. But engineering salaries, oncall burden, and infrastructure maintenance are real costs that don't appear on the cloud bill. The true crossover depends on your team size and ops maturity.

### what api pricing reveals about infrastructure

Decode tokens cost 3-5x more than prefill tokens at OpenAI, Anthropic, and Google. This tells you the system is memory-bandwidth-constrained during decode. Context pricing breakpoints at ~200K tokens reveal where providers switch from KV cache fitting in GPU memory to spilling to slower tiers. Cache hit discounts (50-90% cheaper at some providers) reveal the storage-vs-recompute tradeoff: it's cheaper to keep your KV cache warm than to recompute it.

If you read the pricing page of any inference provider, you're reading a map of their hardware constraints.

---

### observability

**System metrics** (Prometheus + Grafana): GPU utilization, VRAM usage, request queue depth, batch size.

**Model metrics**: TTFT (P50/P90/P99), TPOT (P50/P90/P99), goodput, token counts per request, error rate by type.

**Drift detection** (Evidently AI): KS test or chi-squared test on input distributions. Alert when input patterns shift from the training distribution.

Per-request logging: stamp every request with TTFT, TPOT, token counts, model version, cache hit/miss. You need this for debugging production issues.

### the software stack

Four layers sit between your model and the GPU:

| Layer | Examples | Role |
|-------|----------|------|
| GPU Hardware | H100, B200 | Raw compute and memory |
| CUDA / Triton | CUDA C++, Triton Python | GPU kernel programming |
| Framework | PyTorch, torch.compile | Model definition, optimization |
| Inference Engine | vLLM, SGLang, TensorRT-LLM | Serving: batching, caching, scheduling |
| Orchestration | NVIDIA Dynamo | Multi-node routing, autoscaling |

**Inference engines compared:**

| Engine | Creator | Strengths | Best For |
|--------|---------|-----------|----------|
| **vLLM** | UC Berkeley | PagedAttention, broad model support | General LLM serving |
| **SGLang** | UC Berkeley | RadixAttention, structured generation | Agentic workloads |
| **TensorRT-LLM** | NVIDIA | Maximum NVIDIA hardware utilization | Performance-critical deployments |
| **llama.cpp** | Georgi Gerganov | CPU inference, GGUF quantization | Edge, local inference |

**Model formats**: safetensors (Hugging Face standard, memory-mappable, no arbitrary code execution), GGUF (llama.cpp, includes quantized weights in one file), ONNX (vendor-neutral, cross-platform).

---

## long context and edge

### the long context problem

Standard attention scales O(N^2) with sequence length. At 128K tokens, a single attention layer processes 16 billion token pairs.

**Ring Attention**: distribute the sequence across D devices in a ring. Each device holds a chunk of Q. K, V chunks rotate around the ring. At each step, each device computes local attention and passes its K, V chunk to the next device. After D steps, every device has attended to the full sequence. Memory per device: O(N/D). Context length scales linearly with GPU count.

**RoPE extensions**: Rotary Position Embeddings encode position information. Extending beyond training length requires adjusting the frequency basis. YaRN and NTK-aware scaling do this without fine-tuning.

### edge inference

Running inference on the end user's device. Zero network latency, no server cost, full privacy.

| Runtime | Platform | Key Feature |
|---------|----------|-------------|
| **llama.cpp** | Cross-platform | k-quant formats, GGUF, minimal deps |
| **Apple MLX** | Apple Silicon | Unified memory, native Metal |
| **ONNX Runtime** | Cross-platform | Vendor-neutral, hardware backends |
| **ExecuTorch** | Mobile | PyTorch-native, hardware delegates |

**k-quants** (llama.cpp): mixed-precision quantization where different weight components get different bit widths. Q4_K_M (4-bit with important matrices at higher precision) is the sweet spot for quality vs size on consumer hardware.

---

## what's next

**Diffusion LLMs.** Generate all tokens in parallel over T denoising steps. A 1000-token response takes the same number of steps as a 10-token response. If T < output_length, diffusion beats autoregressive. Open problem: text is discrete (tokens), not continuous (pixels). Adapting diffusion to discrete spaces requires either embedding tokens into continuous space or developing discrete diffusion processes.

**Multi-token prediction.** Train the model to predict 2-4 tokens per forward pass using multiple output heads. Each step produces multiple tokens. Combines with speculative decoding: the model's own predictions serve as the draft.

**Mixture of Depths.** Standard transformers process every token through every layer. MoD adds a learned router that skips layers for easy tokens. Compute scales with input difficulty, not just sequence length.

**Hardware frontiers.** Groq LPU: SRAM-only architecture (no HBM), all model weights in on-chip SRAM. Eliminates the memory bandwidth bottleneck for decode. Limited by total SRAM capacity (~230MB).

Inference cost has dropped sharply over the past two years through hardware improvements, quantization, architectural innovations, and serving optimizations. The exact rate varies by model size and workload, but the trajectory is steep enough that "too expensive to run" today often becomes viable within a year.

---

## hardware comparison

| GPU | VRAM | HBM BW | FP16 TFLOPS | FP8 TFLOPS | NVLink |
|-----|------|--------|-------------|------------|--------|
| H100 SXM | 80GB | 3.35 TB/s | 990 | 1979 | 900 GB/s |
| H200 SXM | 141GB | 4.8 TB/s | 990 | 1979 | 900 GB/s |
| B200 SXM | 192GB | 8.0 TB/s | 2250 | 4500 | 1800 GB/s |
| A100 SXM | 80GB | 2.0 TB/s | 312 | - | 600 GB/s |
| RTX 4090 | 24GB | 1.0 TB/s | 330 | 660 | PCIe |
| M3 Ultra | 192GB | 0.8 TB/s | ~27 | - | Unified |
| Groq LPU | 230MB SRAM | 80 TB/s* | 188 INT8 | - | GroqLink |

*Groq's SRAM-only design achieves extreme bandwidth within a small memory capacity.

---

## end-to-end walkthrough

Deploy a model, measure it, optimize step by step, measure again. Each step changes one variable so you see exactly what moved.

**Setup**: Qwen3-8B on a single RTX 4090 (24GB). Same model, same prompts, same measurement function throughout.

```bash
pip install vllm
```

```python
import time
from vllm import LLM, SamplingParams

SYSTEM = "You are a helpful coding assistant. Answer concisely."
PROMPTS = [f"{SYSTEM}\n\nUser: {q}\nAssistant:" for q in [
    "Write a Python quicksort.",
    "Explain Big-O notation.",
    "What is a hash table?",
    "Describe TCP vs UDP.",
    "Write fizzbuzz in Rust.",
    "What is a mutex?",
    "Explain CAP theorem.",
    "Write binary search in Go.",
]]
PARAMS = SamplingParams(temperature=0.7, max_tokens=256)

def bench(llm, tag, n_runs=3):
    """Measure tok/s and TTFT over n_runs, report averages."""
    total_tokens, total_time = 0, 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = llm.generate(PROMPTS, PARAMS)
        elapsed = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tokens += toks
        total_time += elapsed
    avg_tps = total_tokens / total_time
    avg_ttft = (total_time / (n_runs * len(PROMPTS))) * 1000  # ms per request
    print(f"[{tag}] {avg_tps:.0f} tok/s | {avg_ttft:.0f} ms/req avg")
    return avg_tps
```

### step 0: BF16 baseline

Load the full-precision model. This is what you get out of the box.

```python
llm = LLM(model="Qwen/Qwen3-8B", max_model_len=4096, gpu_memory_utilization=0.90)
bench(llm, "BF16 baseline")
# [BF16 baseline] ~35 tok/s | ~285 ms/req avg
```

16GB of weights. Each decode step reads all 16GB from VRAM to compute one token per sequence. Memory bandwidth is the bottleneck.

### step 1: AWQ quantization (4-bit)

One change: swap BF16 weights for AWQ 4-bit. Model size drops from 16GB to ~4.5GB.

```python
del llm  # free VRAM
llm = LLM(
    model="Qwen/Qwen3-8B-AWQ",
    quantization="awq",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)
bench(llm, "AWQ 4-bit")
# [AWQ 4-bit] ~80 tok/s | ~125 ms/req avg
```

**Why it helps**: decode reads 4.5GB instead of 16GB per step. Same bandwidth, fewer bytes, more tokens per second. The freed 11.5GB of VRAM is now available for KV cache - you can serve more concurrent requests.

### step 2: prefix caching

One change: enable prefix caching. All 8 prompts share the same system prompt, so vLLM caches those prefill tokens and reuses them.

```python
del llm
llm = LLM(
    model="Qwen/Qwen3-8B-AWQ",
    quantization="awq",
    enable_prefix_caching=True,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)
bench(llm, "AWQ + prefix cache")
# [AWQ + prefix cache] ~85 tok/s | ~95 ms/req avg
```

**Why it helps**: the shared system prompt (32 tokens) is prefilled once, then its KV cache is reused for requests 2-8. TTFT drops because those requests skip redundant prefill. Throughput improvement is modest here because the shared prefix is short - with longer system prompts (500+ tokens), the TTFT savings compound.

### step 3: concurrent batching

One change: send requests concurrently instead of in a single batch. This lets vLLM's continuous batching scheduler fill GPU cycles that single-batch decode leaves idle.

```python
import asyncio
from vllm import AsyncLLM, AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen3-8B-AWQ",
    quantization="awq",
    enable_prefix_caching=True,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
)

async def bench_concurrent():
    llm = AsyncLLM.from_engine_args(engine_args)

    async def gen_one(prompt, rid):
        full = []
        async for out in llm.generate(prompt, PARAMS, request_id=f"r{rid}"):
            full.append(out)
        return sum(len(o.outputs[0].token_ids) for o in full[-1:])

    t0 = time.perf_counter()
    results = await asyncio.gather(*[gen_one(p, i) for i, p in enumerate(PROMPTS)])
    elapsed = time.perf_counter() - t0
    total_toks = sum(results)
    print(f"[concurrent] {total_toks/elapsed:.0f} tok/s | {elapsed*1000/len(PROMPTS):.0f} ms/req avg")

asyncio.run(bench_concurrent())
# [concurrent] ~160 tok/s | ~50 ms/req avg
```

**Why it helps**: with 8 concurrent requests, the scheduler batches decode steps together. One weight-read produces 8 tokens instead of 1, pushing closer to the GPU's bandwidth ceiling. This is the same principle as database connection pooling - amortize fixed costs across parallel work.

### results summary

| Step | Change | tok/s | ms/req | Speedup |
|------|--------|-------|--------|---------|
| 0 | BF16 baseline | ~35 | ~285 | 1x |
| 1 | AWQ 4-bit | ~80 | ~125 | 2.3x |
| 2 | + prefix caching | ~85 | ~95 | 2.4x |
| 3 | + concurrent batching | ~160 | ~50 | 4.6x |

Each step changed one variable. The biggest single win was quantization (2.3x) because it directly reduces the memory-bandwidth bottleneck. Concurrent batching gave the next largest jump by amortizing weight reads across requests.

*Numbers are representative for an RTX 4090 with Qwen3-8B. Run the benchmarks on your hardware - the ratios hold across GPUs, but absolute values scale with memory bandwidth.*

### production benchmarking

Once you deploy behind a server (vLLM or SGLang), use genai-perf to measure under realistic load:

```bash
pip install genai-perf

genai-perf profile \
    -m Qwen/Qwen3-8B-AWQ \
    --endpoint-type chat \
    --url localhost:8000 \
    --concurrency 16 \
    --input-tokens-mean 512 \
    --output-tokens-mean 128
```

This reports P50/P90/P99 for TTFT, TPOT, throughput, and goodput. Run it before and after each optimization to measure real impact - the `bench()` function above gives directional signal, but genai-perf simulates production traffic patterns.

---

## further reading

Key papers:

- [**Dao et al., 2022**](https://arxiv.org/abs/2205.14135) - FlashAttention (tiling + online softmax)
- [**Kwon et al., 2023**](https://arxiv.org/abs/2309.06180) - PagedAttention / vLLM (OS virtual memory for KV cache)
- [**Leviathan et al., 2023**](https://arxiv.org/abs/2211.17192) - Speculative decoding (rejection sampling proof)
- [**Xiao et al., 2023**](https://arxiv.org/abs/2211.10438) - SmoothQuant (activation-to-weight difficulty migration)
- [**Lin et al., 2023**](https://arxiv.org/abs/2306.00978) - AWQ (activation-aware weight quantization)
- [**Frantar et al., 2023**](https://arxiv.org/abs/2210.17323) - GPTQ (Hessian-based post-training quantization)
- [**Xiao et al., 2024**](https://arxiv.org/abs/2309.17453) - StreamingLLM (attention sinks for infinite context)
- [**Gu & Dao, 2024**](https://arxiv.org/abs/2312.00752) - Mamba (linear-time sequence modeling)
- [**DeepSeek-AI, 2024**](https://arxiv.org/abs/2412.19437) - DeepSeek-V3 (MLA + MoE at scale)
- [**Li et al., 2024**](https://arxiv.org/abs/2401.15077) - EAGLE (speculative sampling with hidden states)

Tools:

- [vLLM](https://github.com/vllm-project/vllm) - general LLM serving
- [SGLang](https://github.com/sgl-project/sglang) - structured generation + prefix caching
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - local/edge inference
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon inference
- [genai-perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html) - NVIDIA inference benchmarking
