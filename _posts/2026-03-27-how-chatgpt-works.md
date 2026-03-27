---
layout: post
title:  "how chatgpt works"
date:   2026-03-27 00:00:00 +0530
categories: [tech]
tokens: "~5k"
description: "from raw internet text to a streaming response in your browser"
---

Most explanations of ChatGPT either stop at "it predicts the next word" (useless) or require a textbook (also useless). This one goes through the actual pipeline. From raw internet data to a streaming response - pretraining, alignment, inference. Code included.

---

## The Data

Before there's a model, there's data. A lot of it.

GPT-4 was trained on roughly 13 trillion tokens. Most of that starts as raw HTML from [Common Crawl](https://commoncrawl.org/) - 2.7 billion web pages, 200-400TB per crawl. That's mostly garbage.

The cleaning pipeline:
1. Parse HTML, strip tags, extract text
2. Deduplicate - the same article appears on hundreds of mirrors
3. Language filter - Common Crawl has 100+ languages
4. Quality filter - remove spam, adult content, low-quality text

[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (open dataset from HuggingFace) shows this at scale: they start from Common Crawl and end up with 44TB of cleaned text (15 trillion tokens) after all filtering. About 10% of what they scraped.

The model can only learn what's in the training data. No architecture cleverness compensates for bad data. GPT-2 used Reddit upvotes as a quality signal. Modern datasets use classifier-based filters.

![FineWeb data pipeline](/assets/images/chatgpt/fineweb.png){: loading="lazy"}

---

## Tokenization

The model doesn't read text - it reads tokens. A token is usually a word or part of a word. "Unbelievable" might be one token or "un" + "believable", depending on the tokenizer.

Characters are too granular (4096 chars = 4096 steps). Words break on anything outside the training vocabulary. The standard approach is **BPE (Byte Pair Encoding)**: start with bytes, repeatedly merge the most frequent adjacent pair.

Two phases:

**Training** (runs once on the corpus to learn the merge rules):

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Run this ~50,000 times on a large corpus
# Each iteration: find most frequent pair, merge it into a new token
# "l","o","w" -> "lo","w" -> "low"  (if "low" is frequent enough)
```

**Inference** (apply learned merges to new text):

```python
def encode(text, merges):
    tokens = list(text.encode("utf-8"))   # start as raw bytes
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # apply the merge that was learned earliest (lowest priority = learned first)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens
```

Token efficiency matters at inference: fewer tokens per sentence = less compute per request.

![Tokenization overview](/assets/images/chatgpt/tokenisation.png){: loading="lazy"}

<div class="token-animation">
  <p class="anim-label">Tokenizing: "How does ChatGPT work?"</p>
  <div class="token-row">
    <span class="token t1">How</span>
    <span class="token t2"> does</span>
    <span class="token t3"> Chat</span>
    <span class="token t4">G</span>
    <span class="token t5">PT</span>
    <span class="token t6"> work</span>
    <span class="token t7">?</span>
  </div>
  <p class="token-count">7 tokens</p>
</div>

<style>
.token-animation {
  background: #1a1a2e;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
  font-family: monospace;
}
.anim-label {
  color: #888;
  font-size: 13px;
  margin: 0 0 14px 0;
}
.token-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.token {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 15px;
  font-weight: 600;
  opacity: 0;
  transform: translateY(8px);
  animation: fadeUp 0.3s forwards;
}
.token-count {
  color: #666;
  font-size: 12px;
  margin: 12px 0 0 0;
  opacity: 0;
  animation: fadeUp 0.3s 2.5s forwards;
}
.t1 { background: #3b4a6b; color: #a8c7fa; animation-delay: 0.2s; }
.t2 { background: #4a3b6b; color: #c4a8fa; animation-delay: 0.5s; }
.t3 { background: #3b6b4a; color: #a8fac4; animation-delay: 0.8s; }
.t4 { background: #6b4a3b; color: #fac4a8; animation-delay: 1.1s; }
.t5 { background: #6b3b4a; color: #faa8c4; animation-delay: 1.4s; }
.t6 { background: #4a6b3b; color: #c4faa8; animation-delay: 1.7s; }
.t7 { background: #3b6b6b; color: #a8fafa; animation-delay: 2.0s; }
@keyframes fadeUp {
  to { opacity: 1; transform: translateY(0); }
}
</style>

---

## The Transformer

Modern LLMs use the decoder-only transformer: a stack of identical blocks that take token sequences and output a probability distribution over the next token.

Three parts:
1. **Embedding layer**: token IDs to dense vectors (4096-dimensional for large models)
2. **N transformer blocks**: 32-96 of them, refine the representations
3. **Output layer**: final hidden state to logit vector (one float per vocabulary token)

Each block has two sublayers:
- **Attention**: tokens communicate with each other
- **MLP**: each token independently processes its own representation

Attention is the key piece. Every token projects itself into three vectors:
- **Q (Query)**: what am I looking for?
- **K (Key)**: what do I advertise about myself?
- **V (Value)**: what information do I carry?

The attention score between positions i and j is `dot(Q_i, K_j)`. Scale by `sqrt(d_k)` for numerical stability, softmax to get weights, multiply by V to get the output.

```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)               # (seq_len, seq_len)
    if mask is not None:
        scores = scores + mask                      # -inf for future positions
    scores -= scores.max(axis=-1, keepdims=True)   # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

The causal mask fills future positions with -inf before softmax, so each token only attends to itself and earlier tokens. This constraint is what makes generation possible: you can produce output one token at a time without peeking ahead.

**Multi-head attention**: run 8-16 attention operations in parallel with different weight matrices. Different heads learn different relationships - subject-verb agreement, coreference, syntactic structure. Concatenate and project.

![Transformer architecture](/assets/images/chatgpt/archTrans.png){: loading="lazy"}

<div class="attn-viz">
  <p class="anim-label">Self-attention: "The cat sat on the mat"</p>
  <div class="attn-tokens">
    <span class="atk" id="atk0">The</span>
    <span class="atk" id="atk1">cat</span>
    <span class="atk" id="atk2">sat</span>
    <span class="atk" id="atk3">on</span>
    <span class="atk" id="atk4">the</span>
    <span class="atk" id="atk5">mat</span>
  </div>
  <p class="attn-caption">Each token attends to previous tokens. "sat" attends most to "cat" (subject of verb).</p>
</div>

<style>
.attn-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.attn-tokens {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.atk {
  padding: 6px 14px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 14px;
  font-weight: 600;
  border: 1px solid #2a4a6b;
  color: #a8c7fa;
  background: #1a2a3b;
  opacity: 0;
  animation: fadeUp 0.25s forwards;
}
#atk0 { animation-delay: 0.1s; }
#atk1 { animation-delay: 0.3s; background: #1a3b2a; border-color: #2a6b4a; color: #a8fac4; }
#atk2 { animation-delay: 0.5s; background: #3b2a1a; border-color: #6b4a2a; color: #fac4a8; }
#atk3 { animation-delay: 0.7s; }
#atk4 { animation-delay: 0.9s; background: #1a2a3b; border-color: #2a4a6b; color: #a8c7fa; }
#atk5 { animation-delay: 1.1s; background: #2a1a3b; border-color: #4a2a6b; color: #c4a8fa; }
.attn-caption {
  color: #666;
  font-size: 12px;
  margin: 12px 0 0 0;
  opacity: 0;
  animation: fadeUp 0.3s 1.5s forwards;
}
</style>

---

## Pretraining

Take 15 trillion tokens of cleaned text. Feed sequences into the transformer. At each position, predict the next token. Cross-entropy loss against the actual token. Backprop. Repeat.

That's pretraining. No labels, no human annotation. The supervision signal is the text itself.

The model learns whatever is predictable in human-written text: grammar, facts, reasoning patterns, code structure, mathematical notation. It's forced to learn the structure of language to predict well.

GPT-2 as a concrete example:
- Dataset: WebText - 40GB, 8 million Reddit-linked documents
- Model: 1.5B parameters, 48 layers, 1024 token context window
- Cost: roughly $50,000 in 2019, about 1 week on V100 GPUs

GPT-4 scale: trillions of tokens, thousands of GPUs, months. The exact numbers aren't public, but training runs at this scale cost $50-100M+.

What comes out: a very good autocomplete. Feed it "The capital of France is" - it outputs "Paris" with high probability. Feed it a half-written function - it completes it. But ask it a question and it'll continue your text, not answer it. The base model has no concept of "question" vs "answer". It's learned text continuation.

Pretraining is where the money goes. SFT and alignment together cost less than a single day of pretraining compute.

---

## Post-Training: Making It an Assistant

Two stages to go from base model to ChatGPT.

**SFT (Supervised Fine-Tuning)**

Curate 10,000-100,000 examples of (prompt, ideal response). Train the model on these with the same loss function, much smaller learning rate. The model learns to produce responses shaped like your examples.

SFT is cheap because the knowledge is already in the weights - you're steering, not teaching. The base model knows what "the capital of France" is; SFT teaches it to respond in a useful format instead of just continuing the text.

Problem: "ideal" is subjective. Different annotators write different ideal responses. The loss function treats every output token equally - there's no signal for tone, helpfulness, or avoiding harmful content.

**RLHF (Reinforcement Learning from Human Feedback)**

1. Generate 4-8 responses to each prompt from the SFT model
2. Human annotators rank them: A > C > B > D
3. Train a **reward model** - a neural network that learns to predict human preference scores from these rankings
4. Use PPO to update the SFT model to maximize reward model scores, with a KL-divergence penalty to stop it from drifting too far from the SFT starting point

The reward model captures what's hard to specify in advance: tone, helpfulness, not hedging when you should just answer. Human preferences are implicit; ranking is how you extract them.

Pretraining gives capability. Post-training shapes behavior. RLHF doesn't teach new facts - it reshapes how the model uses what it already knows.

![SFT data preparation](/assets/images/chatgpt/data_prep_sft.png){: loading="lazy"}

![RLHF/PPO training loop](/assets/images/chatgpt/ppo.png){: loading="lazy"}

**Thinking models**

o1, DeepSeek-R1: same RLHF idea, but RL-trained to emit reasoning tokens before the final answer. The model produces a `<think>...</think>` block - working through the problem - before committing to a response. Helps for math and code. Overkill for "what's the capital of France?"

---

## Decoding: How Text Gets Generated

The model outputs logits - a vector of ~50,000 floats, one per vocabulary token. Higher logit = model thinks this is more likely next. How do you pick a token?

**Greedy**: always pick the highest-probability token. Deterministic. Repetitive and boring.

**Temperature**: scale logits before softmax. Temperature < 1.0 sharpens the distribution (more confident). Temperature > 1.0 flattens it (more random). Temperature approaching 0 approaches greedy.

**Top-P (nucleus sampling)** - what ChatGPT uses: pick the smallest set of tokens whose cumulative probability exceeds p=0.9. Sample from that set.

```python
import numpy as np

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    # softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    # sort descending by probability
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    # find cutoff: smallest set with cumulative prob >= p
    cumsum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumsum, p)) + 1
    # sample from the nucleus
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs /= nucleus_probs.sum()
    chosen_rank = np.random.choice(cutoff, p=nucleus_probs)
    return int(sorted_idx[chosen_rank])
```

The generation loop: forward pass, get logits, sample token, append to sequence, forward pass again. Repeat until end-of-sequence token or max_tokens limit.

![Top-P sampling distribution](/assets/images/chatgpt/topp.png){: loading="lazy"}

<div class="topp-viz">
  <p class="anim-label">Top-P sampling (p=0.9): probability distribution over next token</p>
  <div class="bar-container">
    <div class="bar-wrap">
      <div class="bar b1" style="--h:85%"><span class="bar-label">Paris</span></div>
    </div>
    <div class="bar-wrap">
      <div class="bar b2" style="--h:6%"><span class="bar-label">Lyon</span></div>
    </div>
    <div class="bar-wrap">
      <div class="bar b3" style="--h:4%"><span class="bar-label">France</span></div>
    </div>
    <div class="bar-wrap">
      <div class="bar b4 excluded" style="--h:2%"><span class="bar-label">Nice</span></div>
    </div>
    <div class="bar-wrap">
      <div class="bar b5 excluded" style="--h:1%"><span class="bar-label">Marseille</span></div>
    </div>
    <div class="bar-wrap">
      <div class="bar b6 excluded" style="--h:0.5%"><span class="bar-label">...</span></div>
    </div>
  </div>
  <p class="topp-note">Top 3 tokens cover ~95% cumulative probability. Nucleus = {Paris, Lyon, France}. Sample from these only.</p>
</div>

<style>
.topp-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.bar-container {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  height: 120px;
  margin: 16px 0 8px;
}
.bar-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
}
.bar {
  width: 100%;
  height: var(--h);
  border-radius: 4px 4px 0 0;
  position: relative;
  animation: growUp 0.5s ease-out both;
  transform-origin: bottom;
}
.bar-label {
  position: absolute;
  bottom: -20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: #888;
  white-space: nowrap;
  font-family: monospace;
}
.b1 { background: #2a6b4a; animation-delay: 0.1s; }
.b2 { background: #2a4a6b; animation-delay: 0.2s; }
.b3 { background: #4a2a6b; animation-delay: 0.3s; }
.excluded { background: #2a2a2a; }
.b4 { animation-delay: 0.4s; }
.b5 { animation-delay: 0.5s; }
.b6 { animation-delay: 0.6s; }
.topp-note {
  color: #666;
  font-size: 12px;
  margin: 28px 0 0 0;
  opacity: 0;
  animation: fadeUp 0.3s 1.2s forwards;
}
@keyframes growUp {
  from { transform: scaleY(0); opacity: 0; }
  to { transform: scaleY(1); opacity: 1; }
}
</style>

---

## Production Inference

What happens when you press Enter on chat.openai.com:

1. `POST /v1/chat/completions` - messages, model name, temperature, max_tokens
2. CDN: TLS termination, geo-routing (~20ms)
3. API gateway: API key check (Redis lookup), rate limiting (token bucket), request validation (~5ms)
4. Backend: builds full context (system prompt + conversation history + your message), tokenizes it, checks context window (~10ms)
5. Inference cluster: 99% of cost and latency

**The two phases**

*Prefill*: process your entire input in one shot. The transformer attends over all input tokens simultaneously. Outputs: the KV cache (key-value matrices for every layer, every input token). Compute-bound.

*Decode*: generate tokens one at a time. Each step: forward pass for the new token only, using cached K/V from prefill. Append new token's K/V to cache. Sample. Repeat. Memory-bandwidth-bound - for every token you read the entire model from GPU memory.

**KV Cache**

Without caching: at decode step n, recompute K/V for all n previous tokens every step. O(n^2) work.

With caching: compute K/V for the new token only, append. O(n) work. The cache lives in GPU VRAM.

<div class="kv-viz">
  <p class="anim-label">KV cache growing during decode</p>
  <div class="kv-row" id="kvrow">
    <div class="kv-block prefill-block">Prefill: [system + user prompt]</div>
  </div>
  <div class="decode-tokens" id="dtokens"></div>
  <p class="kv-note" id="kvnote"></p>
</div>

<script>
(function() {
  var tokens = ["The", " capital", " of", " France", " is", " Paris", "."];
  var container = document.getElementById("dtokens");
  var note = document.getElementById("kvnote");
  var delay = 800;
  tokens.forEach(function(tok, i) {
    setTimeout(function() {
      var el = document.createElement("span");
      el.className = "kv-token";
      el.textContent = tok;
      el.style.animationDelay = "0s";
      container.appendChild(el);
      if (i === tokens.length - 1) {
        setTimeout(function() {
          note.textContent = "Cache size grows with each token. Longer conversations = more VRAM.";
          note.style.opacity = "1";
        }, 400);
      }
    }, delay * (i + 1));
  });
})();
</script>

<style>
.kv-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.kv-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 12px 0 8px;
}
.prefill-block {
  background: #1a3b2a;
  border: 1px solid #2a6b4a;
  color: #a8fac4;
  padding: 5px 12px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 13px;
}
.decode-tokens {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}
.kv-token {
  background: #1a2a3b;
  border: 1px solid #2a4a6b;
  color: #a8c7fa;
  padding: 5px 10px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 13px;
  opacity: 0;
  animation: fadeUp 0.3s forwards;
}
.kv-note {
  color: #666;
  font-size: 12px;
  margin: 10px 0 0 0;
  opacity: 0;
  transition: opacity 0.4s;
}
</style>

**Continuous Batching**

Naive static batching: wait for all requests in a batch to finish before starting new ones. One long request blocks everything.

Continuous batching: at each decode step, check which sequences are done. Fill finished slots with new requests immediately. GPU is always full.

**Quantization**

Default weights are float32 (4 bytes/param). A 7B model = 28GB. Reducing precision cuts memory and speeds up inference:

| Format | Bytes/param | 7B model | Quality loss |
|--------|-------------|----------|--------------|
| FP32 | 4 | 28 GB | - |
| BF16 | 2 | 14 GB | negligible |
| INT8 | 1 | 7 GB | negligible |
| INT4 | 0.5 | 3.5 GB | ~10% |
| INT2 | 0.25 | 1.75 GB | unusable |

16 to 4 bit: the practical sweet spot. Used in llama.cpp, Ollama, any local inference. 16 to 2 bit: the model becomes incoherent.

**Speculative Decoding**

Each decode step requires one full forward pass through the large model, so throughput is limited by how fast you can run it.

Fix: a small draft model (7B) generates 4-5 candidate tokens speculatively. The large model verifies all of them in a single forward pass via rejection sampling - accepts tokens where its distribution matches the draft. Even accepting 3 of 5 is a net win. Works because text is predictable: "The Eiffel Tower is located in" - small model drafts "Paris, France." - large model accepts both in one pass.

**Flash Attention**

Standard attention materializes the full seq x seq attention matrix in slow HBM (GPU's main memory). For a 4096-token sequence with 128-dim heads: 4096 x 4096 = 128MB per head per layer.

Flash Attention tiles the computation. Breaks Q, K, V into blocks, computes attention in tiles that fit in fast SRAM, never writes the full matrix to HBM. Same output, far less memory traffic. For long contexts (16k+ tokens): meaningful speedup.

![System design overview](/assets/images/chatgpt/sysDesign.png){: loading="lazy"}

---

## The Full Picture

From a URL in a web crawl to a token in your browser: crawl the web, clean it, tokenize it, train a transformer to predict next tokens, fine-tune it to follow instructions, optimize its weights for the hardware, and serve it at scale.

The expensive parts: pretraining (trillions of tokens, months on thousands of GPUs) and inference at scale (hundreds of H100s running constantly). The cheap parts: alignment (weeks of fine-tuning plus human annotation) and quantization (run post-hoc, no retraining).

The model doesn't "think". It compresses patterns from training data into weights, then uses those weights to continue token sequences. That's enough to write code, explain concepts, and hold a conversation - but the model has no mechanism to flag when a completion is plausible versus true.
