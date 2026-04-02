---
layout: post
title:  "scaling laws"
date:   2026-01-06 14:06:04 +0530
categories: tech
tokens: "~1.2k"
description: "Maybe intelligence is just a lot of compute applied to a lot of data having a lot of parameters. OpenAI wrote a paper on this."
---

OpenAI wrote a paper on what happens to model performance as you scale parameters, data, and compute. It's called [**Scaling Laws for Neural Language Models**](https://arxiv.org/pdf/2001.08361), 2020.

-----

## scaling laws

![sl](/assets/images/scaling_laws/scaling_laws.webp)

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/scaling/scaling_curves_poster.jpg">
  <source src="/assets/images/scaling/scaling_curves.mp4" type="video/mp4">
</video>
*Scaling parameters alone, data alone, or both (Chinchilla) - same compute, different loss.*

SL: simple predictive laws / rules for behaviour of language model performance.

*HOW WE TEST:*

- OLD: Tune hyper-parameters on big models
- NEW: Tune on small models -> extrapolate to large ones

Scaling Laws Paper says validation and test loss decreases as parameter and number of layers increases along with compute increase.

**IDEA:** Do all experimentations on small model with less compute -> Nail the big model in one-go.

Kaplan et al. (2020) quantified this with three power law equations:

$$L(N) \sim N^{-0.076}$$

$$L(D) \sim D^{-0.095}$$

$$L(C) \sim C^{-0.050}$$

where *N* = number of parameters, *D* = dataset size (tokens), *C* = compute budget (FLOPs), and *L* = cross-entropy loss. Loss decreases as a power law as you add more parameters, data, or compute, but the small exponents mean you get diminishing returns: each 10x increase in any factor yields only a modest drop in loss.

**Maybe intelligence -> lot of compute applied to lot of data having lot of parameters.**

This was the first idea of scaling laws in the 1970s

- Training on enough data matters: GPT-3 was undertrained.
- Chinchilla (half parameter size of GPT3 (70b) but 4x data -> performed better)

<style>
.chinchilla-widget{background:var(--bg-card,#1f2335);border:1px solid var(--border,#e1e4e8);border-radius:10px;padding:20px 24px;margin:24px 0;font-family:inherit}
.chinchilla-widget h4{margin:0 0 12px;color:var(--text-heading,#c0caf5);font-size:16px}
.chinchilla-widget p.desc{font-size:13px;color:var(--text-secondary,#787c99);margin:0 0 16px}
.chinchilla-widget .presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.chinchilla-widget .presets button{background:var(--bg-code,#16161e);color:var(--text,#a9b1d6);border:1px solid var(--border,#e1e4e8);border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px;font-family:var(--font-mono,'JetBrains Mono',monospace);transition:background .15s,border-color .15s}
.chinchilla-widget .presets button:hover,.chinchilla-widget .presets button.active{background:var(--accent,#0366d6);color:#fff;border-color:var(--accent,#0366d6)}
.chinchilla-widget input[type=text]{width:100%;box-sizing:border-box;background:var(--bg-input,#1f2335);color:var(--text,#a9b1d6);border:1px solid var(--border,#e1e4e8);border-radius:6px;padding:8px 12px;font-size:14px;font-family:var(--font-mono,'JetBrains Mono',monospace);margin-bottom:16px}
.chinchilla-widget .results{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.chinchilla-widget .result-card{background:var(--bg-code,#16161e);border-radius:8px;padding:14px 16px;border-left:3px solid var(--accent,#0366d6)}
.chinchilla-widget .result-card.old{border-left-color:#ff6b6b}
.chinchilla-widget .result-card .label{font-size:12px;color:var(--text-muted,#565f89);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
.chinchilla-widget .result-card .val{font-size:15px;font-weight:600;color:var(--text-heading,#c0caf5);font-family:var(--font-mono,'JetBrains Mono',monospace)}
.chinchilla-widget .result-card .val span{font-weight:400;font-size:12px;color:var(--text-muted,#565f89)}
.chinchilla-widget .summary{margin-top:14px;padding:12px 16px;background:var(--bg-code,#16161e);border-radius:8px;font-size:13px;color:var(--text,#a9b1d6);line-height:1.6;font-family:var(--font-mono,'JetBrains Mono',monospace)}
@media(max-width:500px){.chinchilla-widget .results{grid-template-columns:1fr}}
</style>

<div id="chinchilla-calc" class="chinchilla-widget">
  <h4>Chinchilla Compute Allocator</h4>
  <p class="desc">Enter a compute budget in FLOPs. Compare Kaplan (2020) allocation vs Chinchilla (2022) optimal allocation.</p>
  <div class="presets"></div>
  <input type="text" id="cc-input" placeholder="e.g. 1e21">
  <div class="results">
    <div class="result-card old">
      <div class="label">Kaplan (2020) - params heavy</div>
      <div class="val" id="cc-kaplan">-</div>
    </div>
    <div class="result-card">
      <div class="label">Chinchilla (2022) - balanced</div>
      <div class="val" id="cc-chinchilla">-</div>
    </div>
  </div>
  <div class="summary" id="cc-summary"></div>
</div>

<script>
lazyWidget('chinchilla-calc', function(){
  var presets = [1e18, 1e20, 1e22, 1e24];
  var presetContainer = document.querySelector('#chinchilla-calc .presets');
  var input = document.getElementById('cc-input');
  var kaplanEl = document.getElementById('cc-kaplan');
  var chinchillaEl = document.getElementById('cc-chinchilla');
  var summaryEl = document.getElementById('cc-summary');

  function fmt(n){
    if(n>=1e15) return (n/1e15).toFixed(1)+'Q';
    if(n>=1e12) return (n/1e12).toFixed(1)+'T';
    if(n>=1e9) return (n/1e9).toFixed(1)+'B';
    if(n>=1e6) return (n/1e6).toFixed(1)+'M';
    if(n>=1e3) return (n/1e3).toFixed(1)+'K';
    return n.toFixed(0);
  }

  function fmtFlops(n){
    var exp = Math.floor(Math.log10(n));
    var mantissa = n / Math.pow(10, exp);
    return mantissa.toFixed(1) + 'e' + exp;
  }

  function compute(C){
    // Chinchilla: equal allocation in log space
    // C ~ 6 * N * D, with N_opt ~ C^0.5, D_opt ~ C^0.5
    // More precisely: N_opt = (C/6)^0.5, D_opt = (C/6)^0.5
    // so that 6*N*D = C
    var chinN = Math.pow(C / 6, 0.5);
    var chinD = Math.pow(C / 6, 0.5);

    // Kaplan: params-heavy allocation
    // Kaplan suggested N scales faster: N ~ C^0.73, D ~ C^0.27
    // Using C = 6*N*D constraint: N = (C/6)^0.73, D = C/(6*N)
    var kapN = Math.pow(C / 6, 0.73);
    var kapD = C / (6 * kapN);

    kaplanEl.innerHTML = 'N = ' + fmt(kapN) + ' params<br>D = ' + fmt(kapD) + ' tokens <span>(params-heavy)</span>';
    chinchillaEl.innerHTML = 'N = ' + fmt(chinN) + ' params<br>D = ' + fmt(chinD) + ' tokens <span>(balanced)</span>';
    summaryEl.innerHTML = 'With C = ' + fmtFlops(C) + ' FLOPs:<br>' +
      'Chinchilla says N = ' + fmt(chinN) + ' params, D = ' + fmt(chinD) + ' tokens<br>' +
      'Kaplan would use ' + fmt(kapN) + ' params but only ' + fmt(kapD) + ' tokens<br>' +
      'Ratio: Chinchilla uses ' + (chinD/kapD).toFixed(1) + 'x more data with ' + (kapN/chinN).toFixed(1) + 'x fewer params';
  }

  presets.forEach(function(p){
    var btn = document.createElement('button');
    btn.textContent = '10^' + Math.log10(p);
    btn.onclick = function(){
      document.querySelectorAll('#chinchilla-calc .presets button').forEach(function(b){b.classList.remove('active')});
      btn.classList.add('active');
      input.value = p.toExponential();
      compute(p);
    };
    presetContainer.appendChild(btn);
  });

  input.addEventListener('input', function(){
    document.querySelectorAll('#chinchilla-calc .presets button').forEach(function(b){b.classList.remove('active')});
    var v = parseFloat(input.value);
    if(!isNaN(v) && v > 0) compute(v);
  });

  compute(1e21);
  input.value = '1e21';
});
</script>

Data Scaling Laws: formula that maps dataset size(n)

Loss and 'n' is linear on a loglog plot

![dl](/assets/images/scaling_laws/data_loss.webp){: loading="lazy"}

**Engineering Data Laws:**

How does data composition affect model performance (not just size) -> data composition only affects offset not slope (You can do data select experiments on a much smaller model)

![datacomposition](/assets/images/scaling_laws/dc.webp){: loading="lazy"}

More questions on data:

- We have finite data, how does repeating examples affect scaling?
        - Up to 4 epochs repeating data is almost as good as new but after that it shows rapidly diminishing returns.
- Given that repeated data is less valuable -> data selection should adapt to scale.
- **Repeat high quality data OR include new data (trade-off)**

### how to design a huge LM

- **Architecture:** LSTM vs Transformer (Transformer loss decreases as we increase parameters (MOE is the only thing better than vanilla transformer))
- **Optimiser:** Adam is much better than SGD as we increase epochs (adaptive learning rate (basically takes steps automatically instead of a fixed size))
- **Depth:** layers >=6 is good, 1 vs 2 layers make a huge difference; after that, performance plateaus.
- **Batch Size:** batch size increase -> gradient steps increase (past certain point -> diminishing returns (bias dominates instead of learning deeper features)
  - Critical BatchSize: minimum number of steps for target loss (compute increase -> steps can stay the same (BS fixed))

### side note

> What even are parameters

A number the model can change during training. Training = adjusting numbers so predictions improve.

**Linear regression**

- 𝑦 = 𝑤 𝑥 + 𝑏

- Parameters:
        - w (weight)
        - b (bias)

- So this model has 2 parameters. More parameters = more freedom to fit data.

---

Simple neural layer
𝑦 = 𝑊 𝑥 + 𝑏

If:

- input dim = 4
- output dim = 3

Then:

- W has 4 × 3 = 12 parameters
- b has 3 parameters
- Total = 15 parameters

For LLMs:

```text
Input embeddings
   │
   ▼
Multi-Head Attention
   │  (Wq, Wk, Wv, Wo)
   ▼
Feedforward Network
   │  (W1, W2)
   ▼
Output
```

Every W matrix is full of parameters.

**When people say “LLaMA-7B has 7 billion parameters” : There are 7 billion trainable numbers inside the model.**
Each one:

- A floating-point value (e.g. 16-bit or 32-bit (how do we optimise this? Quantisation (later on (changing from FP32->FP16 without compromising performance so we use lesser bits to store our embeddings))))
- Learned during training
- Frozen at inference

> Parameters are representational capacity, not intelligence (misconception)

Parameters alone aren't enough. The model also needs enough training data and compute.

If:

- Model is huge
- Data is small

Then:

- Model memorizes
- Poor generalization

Hence: Big models need big data.

> Compute ∝ Parameters ∝ Training ∝ Tokens

### recap

- Scaling laws describe how LLM performance improves predictably as you increase parameters, dataset size, or training compute.
- Scale parameters without also scaling data and compute and you hit diminishing returns. Each factor depends on the others.
- The shift to GPUs let researchers scale both model size and dataset size simultaneously. Transformers made that even more efficient.

You’re always budget-constrained, so you pick which of the three knobs to turn: parameters, data, or compute. You can turn all three - there’s no impossibility theorem - but nobody has infinite money. The Chinchilla paper showed most labs were turning the wrong knob. They were making models too big and not training them on enough data. Chinchilla (70B params, 4x the data of GPT-3) outperformed the much larger GPT-3 by allocating the compute budget more wisely.

> Loss ~ f(parameters, data, compute) - all three matter, but how you balance them matters more.

Different labs make different bets on this tradeoff. ChatGPT scales up parameters and broad data with massive compute - optimizing for generality. DeepSeek goes the other way: fewer parameters, higher-quality data, and more inference-time compute - optimizing for reasoning efficiency. Both obey the same scaling laws, they just allocate their budgets differently.

### limitations and future

But we are going out of data:

- Running out of data (quality of data) -> ways to make data synthetic (deepmind alphago plays against itself and only has synthetic data) 
- Reasoning model (o1) bridges this gap (chain of thought) -> longer o1 thinks better it performs (new paradigm for scaling llms -> reasoning (we need higher compute for this)- > current state of AI)
- Invent new arch? Numerical stability of model should be there (none so far, transformer works only)
![sl](/assets/images/scaling_laws/scaling_loss.webp){: loading="lazy"}
