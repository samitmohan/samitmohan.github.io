---
layout: post
title:  "what happens when you press 'submit' on chatgpt"
date:   2026-03-27 00:00:00 +0530
categories: [tech]
tokens: "~12k"
description: "from raw internet text to a streaming response in your browser - pretraining, alignment, inference, and everything in between"
---

You press Enter. Between that and the first token appearing in your browser: 13 trillion tokens of training data, 96 transformer layers, a reward model trained on human rankings, a KV cache growing token by token in GPU VRAM, and a continuous batching scheduler keeping H100s at 85% utilization. This post covers all of it.

---

## the data

GPT-4 trained on roughly 13 trillion tokens. Most of it starts as raw HTML from [Common Crawl](https://commoncrawl.org/) - 2.7 billion web pages, 200-400TB per crawl, new crawl every two months. Raw Common Crawl is mostly garbage: boilerplate, navigation menus, duplicate content, spam.

![GPT-2 crawler pipeline](/assets/images/chatgpt/gpt2-crawler.png){: loading="lazy"}

GPT-2 tried using Common Crawl directly. The paper notes that large amounts of content was "unintelligible." So they switched to a different signal: Reddit upvotes. Outbound links from posts with 3+ karma. If humans upvoted it, it clears a basic quality bar.

The cleaning pipeline every serious dataset now runs:
1. Parse HTML, strip tags, extract actual content (h1, p tags)
2. Deduplicate - the same article mirrors across hundreds of sites
3. Language filter - Common Crawl covers 100+ languages
4. Quality filter - remove spam, adult content, low-quality text
5. PII removal - strip bank details, API keys, personal information

[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is the best public example of this pipeline. HuggingFace starts from Common Crawl and ends up with 44TB of cleaned text (15 trillion tokens) after all filtering. About 10% of what they scraped survives. They use a blocklist from the University of Toulouse to remove adult content, then language filtering (English only), deduplication, and classifier-based quality filtering.

![FineWeb data pipeline](/assets/images/chatgpt/fineweb.png){: loading="lazy"}

A table with `text`, `id`, `dump`, and `url` columns. Loading it:

```python
from datasets import load_dataset

dataset = load_dataset("allenai/c4", "en", split="train")
print(dataset[0])
# {
#   'url': 'https://klyq.com/beginners-bbq-class...',
#   'text': 'Beginners BBQ Class Taking Place in Missoula!...',
#   'timestamp': '2019-04-25T12:57:54Z'
# }
```

The model learns only what's in the training data. Two models at identical parameter count but different data quality will perform very differently. No architecture change fixes a bad corpus.

---

## tokenization

The model reads integers, not text.

A token is roughly a word or subword. "Unbelievable" might be one token or split into "un" + "believable" depending on frequency in the training corpus. Characters are too granular (4096 characters = 4096 steps). Full words break on anything outside the training vocabulary - you can't handle novel words or code identifiers.

The standard is **BPE (Byte Pair Encoding)**. Start with individual bytes, repeatedly merge the most frequent adjacent pair.

![Tokenization overview](/assets/images/chatgpt/tokenisation.png){: loading="lazy"}

Two phases:

**Training** - runs once on the corpus to learn merge rules:

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

# Run ~50,000 times on a large corpus
# "t","h" -> "th" -> "the" (if "th"+"e" is frequent enough)
```

**Inference** - apply learned merges to new text:

```python
def encode(text, merges):
    tokens = list(text.encode("utf-8"))   # start as raw bytes
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens
```

In practice you'd use tiktoken:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("How does ChatGPT work?")
print(tokens)         # [4438, 1587, 91985, 990, 30]
print(len(tokens))    # 5
```

![Training phase tokenization](/assets/images/chatgpt/trainingPhasetoken.png){: loading="lazy"}

![Inference phase tokenization](/assets/images/chatgpt/inferencetokenisation.png){: loading="lazy"}

You can explore token splits interactively at [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/):

![TikTokenizer playground](/assets/images/chatgpt/tiktokeniser.png){: loading="lazy"}

Fewer tokens per sentence = less compute per request. Token efficiency is a real cost lever at scale.

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

## the transformer

Modern LLMs use a decoder-only transformer. Token integers go in, a probability distribution over the next token comes out.

![Decoder-only architecture](/assets/images/chatgpt/decoderonly.png){: loading="lazy"}

Three parts:
1. **Embedding layer**: token IDs to dense vectors (4096-dimensional for large models)
2. **N transformer blocks**: 32-96 identical blocks, each refining the representations
3. **Output layer**: final hidden state to a logit vector - one float per vocabulary token (~50,000)

![Input and output of decoder](/assets/images/chatgpt/inpoutofdecode.png){: loading="lazy"}

![Embedding and transformer block flow](/assets/images/chatgpt/trans.png){: loading="lazy"}

Each block has two sublayers:
- **Attention**: tokens communicate with each other
- **MLP (Feed-Forward Network)**: each token independently processes its own representation

Attention moves information between tokens. MLP changes each token's representation based on what it collected. Residual connections accumulate - each block adds to the previous representation rather than replacing it.

```
Attention = Routing = Information Movement
MLP/FFN   = Computation = Processing and Inference
Residual  = Memory = Accumulation
```

### how attention works

Every token computes three vectors at every layer:
- **Q (Query)**: what am I looking for?
- **K (Key)**: what do I advertise about myself?
- **V (Value)**: what information do I carry?

Attention score between positions i and j: `dot(Q_i, K_j) / sqrt(d_k)`. Scale by `sqrt(d_k)` for numerical stability. Softmax to get weights. Weighted sum of V gives the output.

Like a search engine: Query is your search string, Keys are document titles, Values are the document contents. Dot product is the relevance score.

For "bank" in "river bank":
- `Q_bank · K_river` = high score, borrows "river" meaning
- `Q_bank · K_finance` = low score, ignores finance context

```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask          # -inf for future positions
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

The causal mask fills future positions with -inf before softmax. Each token only attends to itself and earlier tokens - this constraint makes generation possible one token at a time without peeking ahead.

**Multi-head attention**: run 8-16 attention operations in parallel with different weight matrices. Different heads learn different relationships - subject-verb agreement, coreference, syntactic structure. Concatenate and project back.

For a deeper look at how residual connections and attention interact across layers, see [from residual connections to attention residuals](/tech/2026/03/18/attention-residuals.html).

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
  <p class="attn-caption">"sat" attends most to "cat" (subject of the verb). Different heads learn different dependency types.</p>
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

### context window

The context window is the model's working memory - everything it can attend to at once. Messages that scroll out are gone; the model has no access to them.

GPT-2: 1,024 tokens. GPT-4o: 128,000 tokens. Larger context = more GPU memory and more compute per token, but enables longer documents and conversations.

<div class="ctx-win-viz">
  <p class="anim-label">Context window filling: tokens accumulate, oldest get evicted when full</p>
  <div class="ctx-win-outer">
    <div class="ctx-win-inner" id="ctxwin-bar"></div>
  </div>
  <div class="ctx-win-meta">
    <span id="ctxwin-count">0 tokens used</span>
    <span>128,000 token limit</span>
  </div>
  <p class="ctx-win-note" id="ctxwin-note">Tokens accumulate left to right. When the window fills, oldest tokens are evicted - the model loses access to them entirely.</p>
</div>

<script>
window._anims = window._anims || {};
(function(){
  var timers=[];
  window._anims.ctxwin = function(){
    timers.forEach(clearTimeout); timers=[];
    var bar=document.getElementById("ctxwin-bar");
    var countEl=document.getElementById("ctxwin-count");
    var noteEl=document.getElementById("ctxwin-note");
    if(!bar) return;
    bar.innerHTML='';
    if(countEl) countEl.textContent='0 tokens used';
    if(noteEl){ noteEl.textContent='Tokens accumulate left to right. When the window fills, oldest tokens are evicted.'; noteEl.style.color='#666'; }
    var segs=[
      {label:"system",  flex:5,  color:"#1a2a3b",border:"#2a4a6b",text:"#a8c7fa", tok:6400},
      {label:"turn 1",  flex:12, color:"#1a3b2a",border:"#2a6b4a",text:"#a8fac4", tok:21760},
      {label:"turn 2",  flex:12, color:"#3b3b1a",border:"#6b6b2a",text:"#fafaa8", tok:37120},
      {label:"turn 3",  flex:18, color:"#2a1a3b",border:"#4a2a6b",text:"#c4a8fa", tok:60160},
      {label:"turn 4",  flex:20, color:"#3b1a2a",border:"#6b2a4a",text:"#faa8c4", tok:85760},
      {label:"new msg", flex:8,  color:"#1a3b3b",border:"#2a6b6b",text:"#a8fafa", tok:96000},
      {label:"overflow \u2192",flex:25,color:"#130a0a",border:"#3a1212",text:"#5a2020",tok:128000},
    ];
    segs.forEach(function(seg,i){
      timers.push(setTimeout(function(){
        var el=document.createElement("div");
        el.className="ctx-win-seg";
        el.style.flex=seg.flex+"";
        el.style.background=seg.color;
        el.style.borderColor=seg.border;
        el.style.color=seg.text;
        el.textContent=seg.label;
        bar.appendChild(el);
        if(countEl) countEl.textContent=seg.tok.toLocaleString()+" tokens used";
        if(i===segs.length-1 && noteEl){ noteEl.style.color="#8a3a3a"; noteEl.textContent="Window full. System prompt and turn 1 get truncated - the model has no memory of them."; }
      },600*(i+1)));
    });
  };
})();
</script>

<style>
.ctx-win-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.ctx-win-outer {
  height: 36px;
  background: #070c10;
  border: 1px solid #1a2a3b;
  border-radius: 4px;
  overflow: hidden;
  margin: 12px 0 6px;
  padding: 2px;
}
.ctx-win-inner {
  display: flex;
  gap: 2px;
  height: 100%;
}
.ctx-win-seg {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 3px;
  font-family: monospace;
  font-size: 10px;
  font-weight: 600;
  border: 1px solid;
  white-space: nowrap;
  overflow: hidden;
  padding: 0 3px;
  opacity: 0;
  animation: fadeUp 0.35s forwards;
}
.ctx-win-meta {
  display: flex;
  justify-content: space-between;
  font-family: monospace;
  font-size: 11px;
  color: #555;
}
.ctx-win-note {
  color: #666;
  font-size: 12px;
  margin: 10px 0 0;
  transition: color 0.6s;
}
</style>

### when does generation stop?

Three stopping conditions:
- **EOS token**: model generates a special end-of-sequence token
- **Max tokens**: hits the configured `max_tokens` limit
- **Repetition penalty**: stops if a token is repeated too many times

---

## pretraining

Take 15 trillion tokens of cleaned text. Feed sequences into the transformer. At each position, predict the next token. Cross-entropy loss against the actual token. Backprop. Repeat. (If you want to understand how backprop and the tensor operations underneath this actually work, [building pytorch from scratch](/tech/2026/03/11/minitorch.html) walks through it.)

The supervision signal is the text itself.

The model learns whatever is predictable in human-written text: grammar, facts, reasoning patterns, code structure, mathematical notation. No one programs this in. It falls out of the loss.

**GPT-2** as a concrete example:
- Dataset: WebText - 40GB, 8 million Reddit-linked documents with 3+ upvotes
- Variants: 117M (Small), 345M (Medium), 774M (Large), 1.5B (XL)
- Architecture: decoder-only, 1024 token context window, BPE tokenizer (~50k vocab)
- Cost: ~$43,000-$50,000 in 2019, about a week on V100 GPUs
- Today: you can replicate GPT-2 training on a single 16GB GPU in a few hours with modern code

GPT-4 scale: trillions of tokens, thousands of H100s, months. Training runs at this scale cost $50-100M+. GPT-4 reportedly uses a Mixture of Experts (MoE) architecture - 8 expert FFN layers per block, 2 active per token. Total parameter count ~1.76 trillion, but only ~220 billion activate for any given token. You get the capacity of a 1.76T model at the inference cost of a ~220B model. The exact numbers aren't public.

<div class="moe-viz">
  <p class="anim-label">MoE: router picks 2 of 8 FFN experts per token per layer</p>
  <div class="moe-top">
    <div class="moe-token-box" id="moe-tok-box">token: "Paris"</div>
    <div class="moe-router-lbl">&#8595; router (learned linear layer, softmax scores)</div>
  </div>
  <div class="moe-expert-row">
    <div class="moe-exp" id="mexp0">E0</div>
    <div class="moe-exp" id="mexp1">E1</div>
    <div class="moe-exp" id="mexp2">E2</div>
    <div class="moe-exp" id="mexp3">E3</div>
    <div class="moe-exp" id="mexp4">E4</div>
    <div class="moe-exp" id="mexp5">E5</div>
    <div class="moe-exp" id="mexp6">E6</div>
    <div class="moe-exp" id="mexp7">E7</div>
  </div>
  <p class="moe-note" id="moe-note-txt">Top-2 experts by router score activate. Only ~12.5% of FFN params run per token.</p>
</div>

<script>
window._anims = window._anims || {};
(function(){
  var moeInterval=null, moeIdx=0;
  window._anims.moe = function(){
    if(moeInterval) clearInterval(moeInterval);
    moeIdx=0;
    var examples=[
      {token:'"Paris"',    active:[2,5], note:"Geography domain. E2+E5 score highest. 220B of 1.76T params active."},
      {token:'"def"',      active:[0,7], note:"Code domain. E0+E7 score highest. 220B of 1.76T params active."},
      {token:'"integral"', active:[1,4], note:"Math domain. E1+E4 score highest. 220B of 1.76T params active."},
      {token:'"Hello"',    active:[3,6], note:"Conversational domain. E3+E6 score highest. 220B of 1.76T params active."},
    ];
    function step(){
      var ex=examples[moeIdx%examples.length];
      for(var i=0;i<8;i++){ var el=document.getElementById("mexp"+i); if(el) el.className="moe-exp"; }
      var tb=document.getElementById("moe-tok-box"); if(tb) tb.textContent="token: "+ex.token;
      setTimeout(function(){
        ex.active.forEach(function(e){ var el=document.getElementById("mexp"+e); if(el) el.className="moe-exp moe-active"; });
        var nt=document.getElementById("moe-note-txt"); if(nt) nt.textContent=ex.note;
      },450);
      moeIdx++;
    }
    step();
    moeInterval=setInterval(step,2600);
  };
})();
</script>

<style>
.moe-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.moe-top { text-align: center; margin-bottom: 10px; }
.moe-token-box {
  display: inline-block;
  background: #1a2a3b;
  border: 1px solid #2a4a6b;
  color: #a8c7fa;
  font-family: monospace;
  font-size: 13px;
  font-weight: 700;
  padding: 5px 14px;
  border-radius: 5px;
}
.moe-router-lbl {
  font-family: monospace;
  font-size: 11px;
  color: #3a6b4a;
  margin: 5px 0 2px;
}
.moe-expert-row {
  display: flex;
  justify-content: center;
  gap: 6px;
  flex-wrap: wrap;
}
.moe-exp {
  width: 46px;
  height: 46px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  font-family: monospace;
  font-size: 12px;
  font-weight: 700;
  background: #111820;
  border: 1px solid #1e2a38;
  color: #2a3a4a;
  transition: background 0.35s, border-color 0.35s, color 0.35s, box-shadow 0.35s;
}
.moe-active {
  background: #1a3b2a !important;
  border-color: #2a6b4a !important;
  color: #a8fac4 !important;
  box-shadow: 0 0 10px #1a4a2a80;
}
.moe-note {
  color: #666;
  font-size: 12px;
  margin: 12px 0 0;
  font-family: monospace;
}
</style>

What comes out: a very good autocomplete. Feed it "The capital of France is" and it outputs "Paris" with high probability. Feed it a half-written function and it completes it. Ask it a question and it continues your text instead of answering - the base model has no concept of question vs answer. It learned text continuation. That's all pretraining does. The surprising part is how far that gets you.

### pretraining is compression

Shannon proved that optimal prediction and optimal compression are the same algorithm: a model that assigns probability p to the next token can compress text to -log2(p) bits per token. Better prediction = smaller file.

Marcus Hutter took this seriously. The [Hutter Prize](http://prize.hutter1.net/) offers €500,000 to whoever compresses 1GB of English Wikipedia below a target size. The prize bets that compression = intelligence. To compress English text well, you need to model it: grammar, syntax, world facts, coreference, causal structure. A lookup table won't do it. You need a predictor that generalizes.

This is exactly what pretraining produces. Llama 3 70B achieves roughly 1.0-1.3 bits/character on natural English text. Shannon's 1951 estimate for the entropy of English was ~1 bit/character. Modern LLMs are approaching the theoretical limit of how well you can predict English.

The weights don't store the internet. They store what predicts the internet. That's why the model knows "The Eiffel Tower is in Paris" without having memorized that specific string - it learned the statistical structure that makes "Paris" the right continuation. It's also why the model hallucinates: it's generating the most probable continuation, not retrieving stored facts. When training data is sparse on a topic, plausible-sounding text and true text are indistinguishable to the predictor.

OpenAI's GPT-4 pretraining run reportedly cost over $100M. The alignment phase that turned it into ChatGPT cost a fraction of that - a few weeks of SFT and RLHF on a model that already knew everything.

How much data and how large a model to train? That's the Chinchilla question - covered in detail in [scaling laws](/tech/2026/01/06/scalinglaws.html).

![Pretraining vs post-training overview](/assets/images/chatgpt/preandpost.png){: loading="lazy"}

---

## post-training: making it an assistant

### SFT (supervised fine-tuning)

Curate 10,000-100,000 examples of (prompt, ideal response). Continue training with the same loss function at a much smaller learning rate. The model learns to respond in a useful format.

SFT is cheap. The knowledge is already in the weights; you're redirecting it. The base model knows what "the capital of France" is - SFT teaches it to answer instead of continuing the sentence.

ChatGPT's SFT dataset is InstructGPT: ~14,500 manually curated (instruction, response) pairs written by contractors.

```python
# Same loss as pretraining, just different data
# x = instruction, y = ideal answer
loss = cross_entropy(model(x), y)
```

Problem: "ideal" is subjective. Different annotators write different ideal responses. The loss treats every output token equally - no signal for tone, helpfulness, or avoiding harmful content.

![SFT data preparation](/assets/images/chatgpt/data_prep_sft.png){: loading="lazy"}

### RLHF (reinforcement learning from human feedback)

Code and math have verifiable correct answers. Writing and brainstorming don't - you need a reward model to score them.

Four steps:

**Step 1**: Generate 4-8 responses to each prompt from the SFT model

**Step 2**: Human annotators rank them: A > C > B > D

**Step 3**: Train a reward model from these rankings

```python
# Reward model: (prompt, response) -> scalar score
# Trained on preference pairs: preferred > rejected
# Bradley-Terry model: maximize P(A preferred over B)
```

**Step 4**: Use PPO to update the SFT model to maximize reward model scores, with a KL-divergence penalty to stop it drifting too far from the SFT starting point

![Training reward model](/assets/images/chatgpt/training_reward.png){: loading="lazy"}

![Reward model visualization](/assets/images/chatgpt/reward2.png){: loading="lazy"}

![Scores](/assets/images/chatgpt/scores.png){: loading="lazy"}

![PPO training loop](/assets/images/chatgpt/ppo.png){: loading="lazy"}

<div class="rlhf-loop">
  <p class="anim-label">RLHF training loop</p>
  <div class="rlhf-steps">
    <div class="rstep rs1" id="rs1">SFT Model</div>
    <div class="rs-arrow ra1">&#8595;</div>
    <div class="rstep rs2" id="rs2">Generate<br>4-8 responses</div>
    <div class="rs-arrow ra2">&#8595;</div>
    <div class="rstep rs3" id="rs3">Human ranking<br>A &gt; C &gt; B &gt; D</div>
    <div class="rs-arrow ra3">&#8595;</div>
    <div class="rstep rs4" id="rs4">Train reward model<br>R(prompt, response)</div>
    <div class="rs-arrow ra4">&#8595;</div>
    <div class="rstep rs5" id="rs5">PPO update<br>maximize R + KL penalty</div>
    <div class="rs-arrow ra5">&#8593;</div>
    <div class="rstep rs6" id="rs6">Improved policy</div>
  </div>
</div>

<style>
.rlhf-loop {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.rlhf-steps {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
}
.rstep {
  padding: 8px 18px;
  border-radius: 6px;
  font-family: monospace;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.4;
  opacity: 0;
  animation: fadeUp 0.3s forwards;
}
.rs-arrow {
  color: #3a6b4a;
  font-size: 16px;
  margin-left: 20px;
  opacity: 0;
  animation: fadeUp 0.2s forwards;
}
.rs1 { background: #1a2a3b; color: #a8c7fa; border: 1px solid #2a4a6b; animation-delay: 0.2s; }
.rs2 { background: #1a3b2a; color: #a8fac4; border: 1px solid #2a6b4a; animation-delay: 0.6s; }
.rs3 { background: #3b3b1a; color: #fafaa8; border: 1px solid #6b6b2a; animation-delay: 1.0s; }
.rs4 { background: #2a1a3b; color: #c4a8fa; border: 1px solid #4a2a6b; animation-delay: 1.4s; }
.rs5 { background: #3b1a2a; color: #faa8c4; border: 1px solid #6b2a4a; animation-delay: 1.8s; }
.rs6 { background: #1a3b3b; color: #a8fafa; border: 1px solid #2a6b6b; animation-delay: 2.2s; }
.ra1 { animation-delay: 0.4s; }
.ra2 { animation-delay: 0.8s; }
.ra3 { animation-delay: 1.2s; }
.ra4 { animation-delay: 1.6s; }
.ra5 { animation-delay: 2.0s; color: #6b4a2a; }
</style>

| Stage | Purpose | Data | Objective |
|-------|---------|------|-----------|
| Pretraining | Learn language | Internet text | Next-token prediction |
| SFT | Follow instructions | (instruction, response) pairs | Supervised loss |
| RLHF | Align to human preference | Ranked responses | Maximize reward |

![Summary](/assets/images/chatgpt/summary.png){: loading="lazy"}

### DPO: a simpler alternative to PPO

PPO requires maintaining four models simultaneously: the policy being trained, a frozen reference for the KL penalty, the reward model, and a value function. The optimization loop is fragile and sensitive to hyperparameters.

DPO (Direct Preference Optimization) reframes alignment as supervised learning. Given a preferred response `y_w` and a rejected response `y_l` for the same prompt, the loss is:

```text
loss = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
```

This increases the likelihood of preferred responses relative to the reference policy while decreasing the likelihood of rejected ones. No reward model. No RL loop. The reward is implicit in the policy itself.

Same preference data as RLHF, half the implementation complexity. Llama 3, Mistral, Phi-3, and most open-weight models now use DPO variants (ORPO, SimPO) rather than PPO.

The tradeoff: DPO is less stable on very large preference datasets and more sensitive to data quality. PPO still dominates for reasoning models where RL on verifiable rewards is the primary signal - that's where a proper reward function exists and RL can find solutions humans wouldn't label.

### thinking models

o1, DeepSeek-R1: RL-trained to emit reasoning tokens before the final answer. The model produces a `<think>...</think>` block - working through the problem step by step - before committing to a response.

```text
User: "What is 237 x 194?"

Normal model:
  -> "237 x 194 = 45,978"  (immediate, sometimes wrong)

Thinking model:
  <think>
  237 x 194 = 237 x 200 - 237 x 6
  237 x 200 = 47,400
  237 x 6 = 1,422
  47,400 - 1,422 = 45,978
  </think>
  -> "237 x 194 = 45,978"  (verified via reasoning)
```

**GRPO: How R1 was actually trained**

DeepSeek-R1 uses GRPO (Group Relative Policy Optimization) rather than PPO. The key difference: PPO requires a separate critic network to estimate value baselines. GRPO eliminates it by sampling a group of G responses per prompt and using their relative rewards as the baseline:

```python
# PPO: baseline from a learned value function
advantage = reward - V(state)

# GRPO: baseline is the mean reward of the sampled group
rewards = [r1, r2, r3, r4]   # G=4 responses sampled per prompt
baseline = mean(rewards)
advantages = [r - baseline for r in rewards]
```

For math and code where rewards are binary (correct/incorrect), GRPO is more stable - no value function to train, no actor-critic mismatch. R1's training worked because "does this answer match the ground truth" is a clean reward signal. The structured chain-of-thought emerged from RL on math problems. The model discovered that working through intermediate steps improved its reward on verifiable answers.

**When thinking tokens are worth the cost:**

| Task | Use thinking | Reason |
|------|-------------|--------|
| Multi-step math | Yes | Each step is verifiable, RL can optimize |
| Complex code | Yes | Logic chains benefit from working through |
| Factual lookup | No | Answer is in weights, no reasoning path needed |
| Creative writing | No | Chain-of-thought doesn't help creativity |
| Simple Q&A | No | 10-100x compute overhead, no accuracy gain |

Thinking models are expensive: 500-5000 reasoning tokens before a 50-token answer. For tasks where correctness is verifiable and errors are costly, that overhead is justified. For everything else, the standard model is better value.

---

## decoding: how text gets generated

The model outputs logits - a vector of ~50,000 floats, one per vocabulary token. You need a strategy to pick a token.

<div class="autogen-viz">
  <p class="anim-label">Autoregressive decode: one forward pass per token</p>
  <div class="ag-seq">
    <span class="ag-tok ag-p">The</span><span class="ag-tok ag-p"> capital</span><span class="ag-tok ag-p"> of</span><span class="ag-tok ag-p"> France</span><span class="ag-tok ag-p"> is</span><span id="ag-gen-container"></span><span class="ag-cursor" id="ag-cur">&#9611;</span>
  </div>
  <p class="ag-note" id="ag-note-txt">Prompt processed in prefill. Sampling from 50,000-token distribution...</p>
</div>

<script>
window._anims = window._anims || {};
(function(){
  var timers = [];
  window._anims.autogen = function(){
    timers.forEach(clearTimeout); timers = [];
    var container = document.getElementById("ag-gen-container");
    var noteEl = document.getElementById("ag-note-txt");
    var cursor = document.getElementById("ag-cur");
    if(!container) return;
    container.innerHTML = '';
    if(cursor) cursor.style.display = 'inline';
    if(noteEl) noteEl.textContent = 'Prompt processed in prefill. Sampling from 50,000-token distribution...';
    var tokens=[" Paris","."," It"," is"," known"," as"," the"," City"," of"," Light","."];
    var notes=["' Paris' sampled (p\u224887%). KV cache extended by 1 token.","'.' sampled. Context now 7 tokens.","Sampling continues. Each step reads the full growing KV cache.","","","","","","","","EOS token generated. Decode stops."];
    tokens.forEach(function(tok,i){
      timers.push(setTimeout(function(){
        var span=document.createElement("span");
        span.className="ag-tok ag-g";
        span.textContent=tok;
        container.appendChild(span);
        if(notes[i]) noteEl.textContent=notes[i];
        if(i===tokens.length-1) cursor.style.display="none";
      },700*(i+1)));
    });
  };
})();
</script>

<style>
.autogen-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
  font-family: monospace;
}
.ag-seq {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  margin: 12px 0 8px;
  font-size: 15px;
  line-height: 2;
}
.ag-tok { border-radius: 3px; }
.ag-p { background: #1a2a3b; color: #a8c7fa; padding: 2px 5px; margin-right: 1px; }
.ag-g {
  background: #1a3b2a; color: #a8fac4; border: 1px solid #2a6b4a;
  padding: 2px 6px; margin-left: 1px;
  opacity: 0; animation: fadeUp 0.3s forwards;
}
.ag-cursor { color: #a8c7fa; animation: blink 0.9s infinite; margin-left: 2px; }
.ag-note { color: #666; font-size: 12px; margin: 10px 0 0; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
</style>

![Output probabilities](/assets/images/chatgpt/op.png){: loading="lazy"}

![Choosing from probabilities](/assets/images/chatgpt/choosingprob.png){: loading="lazy"}

### greedy search

Always pick the highest-probability token. Deterministic. Fast. Gets stuck in repetitive loops.

![Greedy search](/assets/images/chatgpt/greedy.png){: loading="lazy"}

### beam search

Keep the top-k most likely sequences at each step (beam width k=3-5). More creative than greedy. More expensive. Used in translation and summarization where you want a few good candidates rather than pure randomness.

![Beam search](/assets/images/chatgpt/beam.png){: loading="lazy"}

### multinomial sampling

Sample according to probabilities. Introduces randomness - you might pick low-probability tokens. Can produce incoherent text if not constrained.

![Multinomial sampling](/assets/images/chatgpt/multinomial.png){: loading="lazy"}

### top-K sampling

Pick the top K tokens by probability. Sample from those K. Fixed candidate set size is a problem: sometimes the model is very confident (K=1 is ideal), sometimes uncertain (K=50 makes sense). A fixed K fits neither case.

![Top-K sampling](/assets/images/chatgpt/topk.png){: loading="lazy"}

### top-P (nucleus) sampling

Pick the smallest set of tokens whose cumulative probability exceeds p (typically 0.9). Candidate set size adapts to the model's confidence.

![Top-P sampling distribution](/assets/images/chatgpt/topp.png){: loading="lazy"}

![Top-P results comparison](/assets/images/chatgpt/topPresults.png){: loading="lazy"}

```python
import numpy as np

def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumsum, p)) + 1
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs /= nucleus_probs.sum()
    chosen_rank = np.random.choice(cutoff, p=nucleus_probs)
    return int(sorted_idx[chosen_rank])
```

### temperature

Scales logits before softmax. Temperature < 1.0 sharpens the distribution (model becomes more confident). Temperature > 1.0 flattens it (more random). Temperature approaching 0 approaches greedy.

```python
# gpt-4o uses these for different tasks:
# code generation: temperature=0.2, top_p=0.95
# creative writing: temperature=1.0, top_p=0.95
# factual Q&A:     temperature=0.0  (greedy)
```

Using Hugging Face:

```python
from transformers import set_seed
set_seed(42)

output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_p=0.92,
    temperature=0.8,
    top_k=0
)
```

Decoding strategy comparison:

| Strategy | Type | When to use |
|---|---|---|
| Greedy | Deterministic | One best answer, formal translation |
| Beam Search | Deterministic | Summarization, a few good candidates |
| Top-K | Stochastic | Chat, creative text, fixed candidate set |
| Top-P | Stochastic | Chat, creative text, dynamic set (modern LLMs) |

![Hyperparameter table](/assets/images/chatgpt/hyperparameter.png){: loading="lazy"}

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

## what happens when you press enter

One request, start to finish.

<div class="pipeline-viz">
  <p class="anim-label">Request flow: Enter key to first token</p>
  <div class="pipeline-steps">
    <div class="pstep ps1">Client<br><span class="ps-sub">POST /v1/chat</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps2">CDN<br><span class="ps-sub">~20ms</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps3">Gateway<br><span class="ps-sub">auth/rate</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps4">Backend<br><span class="ps-sub">tokenize</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps5">Safety<br><span class="ps-sub">~12ms</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps6">GPU Cluster<br><span class="ps-sub">prefill+decode</span></div>
    <div class="ps-arrow">&#8594;</div>
    <div class="pstep ps7">SSE Stream<br><span class="ps-sub">tokens/sec</span></div>
  </div>
</div>

<style>
.pipeline-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
  overflow-x: auto;
}
.pipeline-steps {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: max-content;
}
.pstep {
  padding: 10px 14px;
  border-radius: 6px;
  font-family: monospace;
  font-size: 12px;
  font-weight: 600;
  text-align: center;
  line-height: 1.4;
  opacity: 0;
  animation: fadeUp 0.3s forwards;
}
.ps-sub {
  font-size: 10px;
  font-weight: 400;
  opacity: 0.7;
}
.ps-arrow {
  color: #444;
  font-size: 18px;
  opacity: 0;
  animation: fadeUp 0.2s forwards;
}
.ps1 { background: #1a2a3b; color: #a8c7fa; border: 1px solid #2a4a6b; animation-delay: 0.1s; }
.ps2 { background: #1a3b2a; color: #a8fac4; border: 1px solid #2a6b4a; animation-delay: 0.4s; }
.ps3 { background: #3b2a1a; color: #fac4a8; border: 1px solid #6b4a2a; animation-delay: 0.7s; }
.ps4 { background: #2a1a3b; color: #c4a8fa; border: 1px solid #4a2a6b; animation-delay: 1.0s; }
.ps5 { background: #3b1a1a; color: #faafa8; border: 1px solid #6b2a2a; animation-delay: 1.3s; }
.ps6 { background: #1a3b3b; color: #a8fafa; border: 1px solid #2a6b6b; animation-delay: 1.6s; }
.ps7 { background: #3b3b1a; color: #fafaa8; border: 1px solid #6b6b2a; animation-delay: 1.9s; }
.ps-arrow:nth-child(2)  { animation-delay: 0.25s; }
.ps-arrow:nth-child(4)  { animation-delay: 0.55s; }
.ps-arrow:nth-child(6)  { animation-delay: 0.85s; }
.ps-arrow:nth-child(8)  { animation-delay: 1.15s; }
.ps-arrow:nth-child(10) { animation-delay: 1.45s; }
.ps-arrow:nth-child(12) { animation-delay: 1.75s; }
</style>

![System design overview](/assets/images/chatgpt/sysDesign.png){: loading="lazy"}

| Step | Component | What it does | Latency |
|------|-----------|--------------|---------|
| 1 | Client | POST /v1/chat/completions | - |
| 2 | CDN/Edge | TLS termination, DDoS, geo-routing | ~20ms |
| 3 | API Gateway | Auth, rate limit, route | ~5ms |
| 4 | Backend | Context build, tokenize | ~10ms |
| 5 | Safety | Input moderation classifier | ~12ms |
| 6 | Inference cluster | Prefill + decode | TTFT ~50-200ms, then tokens/sec |
| 7 | Streamer | SSE chunks to client | overlapping with decode |
| 8 | Safety | Output moderation | ~8ms |
| 9 | Post-processing | Detokenize, log, bill | ~2ms |

### 1. client

When you press Enter, the browser sends:

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain transformers."}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

`stream: true` opens a Server-Sent Events connection. Tokens stream back as they're generated. SSE is one-directional (server to client), which is all you need. WebSockets are bidirectional and add unnecessary complexity. SSE works natively over HTTP/2 and browsers reconnect automatically.

### 2. CDN / edge

Cloudflare or Fastly handles TLS termination and DDoS protection at the edge. Geographic routing sends the request to the nearest datacenter. Saves 50-200ms for users far from origin servers.

### 3. API gateway

The front door. Three jobs:

**Auth**: API key validation. Key is hashed and looked up in Redis. Invalid key returns `401` immediately - no traffic goes deeper.

**Rate Limiting** via token bucket:

```text
Token Bucket:
  Rate: 100 tokens/sec
  Bucket capacity: 1000 tokens

  Request arrives: costs 10 tokens -> allowed
  Burst request: costs 500 -> allowed (bucket had space)
  Bucket empty -> REJECTED (429 Too Many Requests)
```

**Routing**: based on model name, route to the appropriate GPU cluster. GPT-4o goes to H100s. GPT-3.5-turbo goes to cheaper A10Gs.

### 4. backend service

Assembles the full context:

```text
System Prompt (fixed per deployment)
    +
Conversation History (retrieved from DB or passed in request)
    +
User's new message
    +
Any tool call outputs
    =
Full Context (up to 128K tokens for GPT-4o)
```

If conversation history is too long, oldest messages get truncated. The backend tokenizes the full context with tiktoken, checks it against the context window limit, and estimates billing.

### 5. safety layer (input)

Before the expensive model sees anything, a fast lightweight classifier checks the prompt for hate speech, self-harm, illegal content, and jailbreak attempts. Runs in milliseconds. If flagged, the request is rejected here - no GPU time wasted on the big model.

### 6. inference cluster

Two distinct phases for every request:

**Prefill**: process the entire input prompt in one forward pass. The transformer attends over all input tokens simultaneously. Output: the KV cache (key-value matrices for every layer, every input token). Compute-bound.

**Decode**: generate tokens one at a time. Each step: forward pass for the new token only, reading cached K/V from prefill. Append new token's K/V to cache. Sample. Repeat. Memory-bandwidth-bound - each token requires loading the entire model from GPU memory.

The GPU cluster runs vLLM, TensorRT-LLM, or SGLang - not raw PyTorch. These handle batching, KV cache management, and kernel optimization.

**Hardware**: NVIDIA A100 (80GB), H100 (80GB), or H200 (141GB). A 100B parameter model at FP16 needs 200GB - that's multiple GPUs. Tensor parallelism splits attention heads across GPUs; pipeline parallelism splits model depth across nodes.

```text
Model Parallelism:
  Node 1: [GPU0] [GPU1] [GPU2] [GPU3]  <- Layers 1-48
  Node 2: [GPU4] [GPU5] [GPU6] [GPU7]  <- Layers 49-96
```

### 7. streaming back

Each token streams back via SSE as it's generated:

```text
data: {"choices":[{"delta":{"content":"The"}}]}
data: {"choices":[{"delta":{"content":" transformer"}}]}
data: {"choices":[{"delta":{"content":" architecture"}}]}
data: [DONE]
```

For a 500-token response at 50 tokens/sec: 10 seconds total. Without streaming you stare at a blank screen for 10 seconds. With streaming you start reading after ~20ms (first token).

### 8. post-processing

After generation: detokenize (token IDs back to text), strip control tokens, normalize whitespace, count tokens for billing.

**Tool Calling / Function Calling**: when the model needs real-time data, it generates a structured JSON tool call instead of text:

```json
{
  "tool_call": {
    "name": "web_search",
    "arguments": {"query": "current weather in Mumbai"}
  }
}
```

The backend intercepts this, runs the tool, injects the result back into context, and the model generates its final answer. That's what makes an AI agent: it takes actions in a loop, not just text in response to text. For how retrieval fits into this (RAG, vector search, chunking), see [rag](/tech/2026/01/27/rag.html).

### output safety layer

Output moderation runs in parallel with streaming, not after it. While the model is generating tokens 10-20, the safety classifier is checking tokens 1-9. If a violation appears mid-stream, the partially-delivered response gets cut and the client receives an error chunk. That parallel design is why output moderation adds only ~8ms of overhead to TTFT rather than to total response time.

The `finish_reason` field in every API response tells you what stopped generation:
- `"stop"` - model generated an EOS token naturally
- `"length"` - hit `max_tokens` limit
- `"content_filter"` - output moderation flagged the response

If you're building on the API and users report truncated responses, `finish_reason` is the first thing to check.

### logging and observability

Every request emits structured metrics at each layer:

```text
request_id:       "req_abc123"
user_id:          "usr_456"
model:            "gpt-4o"
input_tokens:     512
output_tokens:    248
total_latency_ms: 2340
ttft_ms:          210        <- Time To First Token
tps:              87         <- Tokens Per Second during decode
region:           "us-east-1"
gpu_node:         "h100-node-42"
finish_reason:    "stop"
safety_flagged:   false
```

**TTFT (Time to First Token)** and **TPS (Tokens Per Second)** are the two numbers anyone running inference cares about. TTFT measures how quickly you start responding - dominated by prefill and queue wait. TPS measures throughput during decode - dominated by model size, quantization, and GPU memory bandwidth. Optimizing one doesn't automatically improve the other: disaggregated serving exists because prefill and decode respond to different hardware improvements.

### full latency breakdown

For a typical GPT-4o request ("Explain the transformer architecture", ~400 token response):

```text
CDN / TLS handshake              ~20ms
API Gateway (auth + rate limit)  ~5ms
Backend context build            ~10ms
Input safety check               ~12ms
Queue wait (continuous batching) ~30ms
Prefill (12 input tokens)        ~50ms
Decode (400 output tokens)       ~4000ms  (@100 tok/s)
Output safety check              ~8ms
Network streaming                ~15ms
--------------------------------------------
TTFT (Time to First Token):     ~127ms
Total wall-clock:               ~4150ms
```

The bottleneck is decode. Quantization, speculative decoding, and better hardware all target decode throughput.

### cost vs latency tradeoffs

Every production decision is a trade between these three:

| Optimization | Latency | Cost/token | Quality |
|---|---|---|---|
| Larger batch size | higher (more queuing) | lower | unchanged |
| INT4 quantization | lower | lower | slight loss |
| Smaller model (3.5 vs 4) | much lower | much lower | lower |
| More GPUs per request | lower | higher | unchanged |
| Speculative decoding | lower | slightly higher | unchanged |
| Streaming | same total, lower perceived | unchanged | unchanged |
| KV cache reuse | lower | lower | unchanged |

Output tokens cost 2-3x more than input tokens at most providers - the model generates them sequentially while input gets processed in parallel. Cached input (repeated system prompt) is typically half-price. If you're building on the API: back-of-envelope with 500 input + 300 output tokens per request, multiply by volume, then check whether open-weights on your own GPU cluster beats provider pricing at your scale.

---

## inference optimizations

### KV cache

Without caching: at decode step n, recompute K/V for all n previous tokens every step. O(n^2) work.

With caching: compute K/V for the new token only, append. O(n) work per step.

The cache lives in GPU VRAM. Long conversations consume more VRAM because the cache grows with sequence length.

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
          note.textContent = "Cache grows with each new token. Longer context = more VRAM.";
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

**MQA/GQA** - Multi-Query and Grouped-Query Attention: standard multi-head attention (MHA) stores separate K/V for every head, which is expensive in VRAM. Multi-Query Attention (MQA) shares K/V across all heads - smaller cache, but heads can't specialize independently. Grouped-Query Attention (GQA) shares K/V across small groups of heads (e.g., groups of 4) - a practical middle ground. Llama 3, Gemma, Mistral all use GQA.

**PagedAttention (vLLM)**: manages the KV cache like OS virtual memory. Instead of allocating a large contiguous block per sequence (which wastes memory when sequences vary in length), vLLM stores KV in fixed-size pages and maps them with a page table. Allows 2-4x higher batch sizes for the same GPU memory.

### continuous batching

Naive static batching: fill a batch, wait for all requests to finish, start the next batch. One long request holds everyone else.

Continuous batching: at each decode step, check which sequences are done. Fill finished slots immediately with new requests. GPU is always full.

```text
Static batching:
  t=0: [A, B, C] all generating
  t=5: A finishes, GPU sits idle until B and C finish
  t=8: B finishes, still waiting for C
  t=12: C finishes, new batch starts

Continuous batching (vLLM):
  t=0: [A, B, C] all generating
  t=5: A finishes, D joins immediately
  t=8: B finishes, E joins immediately
  GPU never idle
```

Without continuous batching: GPU utilization 20-30%. With it: 80-90%.

<div class="cbatch-viz">
  <p class="anim-label">Static vs continuous batching: GPU slot utilization over time &rarr;</p>

  <p class="cbt-heading">Static batching &mdash; entire batch waits for slowest request</p>
  <div class="cbt-grid">
    <div class="cbt-row"><span class="cbt-lbl">Slot 1</span><div class="cbt-track"><div class="cbt-blk cbt-a" style="flex:1.0">A</div><div class="cbt-idle" style="flex:0.8">idle</div><div class="cbt-blk cbt-d" style="flex:1.0">D</div></div></div>
    <div class="cbt-row"><span class="cbt-lbl">Slot 2</span><div class="cbt-track"><div class="cbt-blk cbt-b" style="flex:1.5">B (long)</div><div class="cbt-idle" style="flex:0.3">idle</div><div class="cbt-blk cbt-e" style="flex:1.0">E</div></div></div>
    <div class="cbt-row"><span class="cbt-lbl">Slot 3</span><div class="cbt-track"><div class="cbt-blk cbt-c" style="flex:0.5">C</div><div class="cbt-idle cbt-waste" style="flex:1.5">wasted &uarr; waiting for B</div></div></div>
  </div>

  <p class="cbt-heading" style="margin-top:16px">Continuous batching &mdash; new request fills slot the moment one finishes</p>
  <div class="cbt-grid">
    <div class="cbt-row"><span class="cbt-lbl">Slot 1</span><div class="cbt-track"><div class="cbt-blk cbt-a cba1" style="flex:1.0">A</div><div class="cbt-blk cbt-d cba2" style="flex:1.0">D</div><div class="cbt-blk cbt-g cba3" style="flex:0.8">G</div></div></div>
    <div class="cbt-row"><span class="cbt-lbl">Slot 2</span><div class="cbt-track"><div class="cbt-blk cbt-b cba1" style="flex:1.5">B</div><div class="cbt-blk cbt-e cba2" style="flex:1.0">E</div><div class="cbt-blk cbt-h cba3" style="flex:0.7">H</div></div></div>
    <div class="cbt-row"><span class="cbt-lbl">Slot 3</span><div class="cbt-track"><div class="cbt-blk cbt-c cba1" style="flex:0.5">C</div><div class="cbt-blk cbt-f cba2" style="flex:1.2">F</div><div class="cbt-blk cbt-i cba3" style="flex:1.0">I</div></div></div>
  </div>
  <p class="cbt-note">No gaps. GPU always working. This is what vLLM implements.</p>
</div>

<style>
.cbatch-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.cbt-heading {
  font-family: monospace;
  font-size: 12px;
  color: #888;
  margin: 0 0 8px 0;
}
.cbt-grid { display: flex; flex-direction: column; gap: 4px; }
.cbt-row { display: flex; align-items: center; gap: 8px; }
.cbt-lbl { font-family: monospace; font-size: 11px; color: #555; width: 44px; flex-shrink: 0; }
.cbt-track { display: flex; flex: 1; gap: 2px; height: 32px; }
.cbt-blk {
  display: flex; align-items: center; justify-content: center;
  border-radius: 4px; font-family: monospace; font-size: 12px; font-weight: 700;
  flex: 1;
}
.cbt-idle {
  display: flex; align-items: center; justify-content: center;
  border-radius: 4px; font-family: monospace; font-size: 10px; font-weight: 400;
  flex: 1;
  background: #0d0d0d; border: 1px dashed #222; color: #333;
}
.cbt-waste { color: #5a2a2a; border-color: #3a1515; background: #130a0a; }
.cbt-a { background: #1a2a3b; border: 1px solid #2a4a6b; color: #a8c7fa; }
.cbt-b { background: #1a3b2a; border: 1px solid #2a6b4a; color: #a8fac4; }
.cbt-c { background: #3b2a1a; border: 1px solid #6b4a2a; color: #fac4a8; }
.cbt-d { background: #2a1a3b; border: 1px solid #4a2a6b; color: #c4a8fa; }
.cbt-e { background: #3b3b1a; border: 1px solid #6b6b2a; color: #fafaa8; }
.cbt-f { background: #3b1a2a; border: 1px solid #6b2a4a; color: #faa8c4; }
.cbt-g { background: #1a3b3b; border: 1px solid #2a6b6b; color: #a8fafa; }
.cbt-h { background: #2a3b1a; border: 1px solid #4a6b2a; color: #c4faa8; }
.cbt-i { background: #3b1a3b; border: 1px solid #6b2a6b; color: #faa8fa; }
.cbt-a, .cbt-b, .cbt-c, .cbt-d, .cbt-e, .cbt-f, .cbt-g, .cbt-h, .cbt-i { }
.cba1 { opacity: 0; animation: fadeUp 0.4s 0.1s forwards; }
.cba2 { opacity: 0; animation: fadeUp 0.4s 0.7s forwards; }
.cba3 { opacity: 0; animation: fadeUp 0.4s 1.3s forwards; }
.cbt-note {
  color: #666; font-size: 12px; margin: 10px 0 0;
  opacity: 0; animation: fadeUp 0.3s 1.8s forwards;
}
</style>

### quantization

Default weights are float32 (4 bytes/param). Reducing precision cuts memory and speeds up inference:

| Format | Bytes/param | 7B model | Quality loss |
|--------|-------------|----------|--------------|
| FP32 | 4 | 28 GB | - |
| BF16 | 2 | 14 GB | negligible |
| INT8 | 1 | 7 GB | negligible |
| INT4 | 0.5 | 3.5 GB | ~10% |
| INT2 | 0.25 | 1.75 GB | unusable |

16 to 4 bit is the practical sweet spot for local inference.

Three approaches to get there:

**PTQ (Post-Training Quantization)**: round weights to INT8/INT4 after training. Fast to apply. Quality can drop on difficult tasks.

**GPTQ/AWQ**: calibrate per-channel or per-group scales to minimize quantization error. Run a small calibration dataset, reconstruct weights layer by layer to minimize the output difference. Better quality than naive rounding, still no retraining needed.

**QLoRA**: store the base model in 4-bit, train low-rank adapters in higher precision. Fine-tune a 65B model on a single 48GB GPU.

```python
from transformers import BitsAndBytesConfig
import torch

# Load in 4-bit with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
)
```

### speculative decoding

Each decode step requires one full forward pass through the large model. At 100 tokens/sec for a 4000ms response, the large model is the bottleneck.

Fix: a small draft model (7B) generates 4-5 candidate tokens speculatively. The large model verifies all of them in a single forward pass via rejection sampling - accepts tokens where its distribution matches the draft. Even 3-of-5 accepted is a net win, because verification is one pass and draft generation is fast.

```text
Draft model (7B, fast):   "The capital of France is Paris."
Large model (70B, slow):  verify all 6 tokens in ONE pass -> all match -> accept
Result: 6 tokens for the cost of 1 large-model forward pass
Speedup: 2-3x with no quality change
```

If a draft token is wrong, the rest get discarded and the large model takes over from that position. Works because text is predictable enough that a good small model gets most tokens right.

<div class="spec-viz">
  <p class="anim-label">Speculative decoding: draft 5, verify 1 pass</p>
  <div class="spec-row">
    <div class="spec-label">Draft (7B):</div>
    <div class="spec-tokens">
      <span class="stoken st1 accepted">The</span>
      <span class="stoken st2 accepted">capital</span>
      <span class="stoken st3 accepted">of</span>
      <span class="stoken st4 accepted">France</span>
      <span class="stoken st5 rejected">is</span>
    </div>
  </div>
  <div class="spec-row" style="margin-top:8px">
    <div class="spec-label">Verify (70B):</div>
    <div class="spec-tokens">
      <span class="stoken st6 accepted">The</span>
      <span class="stoken st7 accepted">capital</span>
      <span class="stoken st8 accepted">of</span>
      <span class="stoken st9 accepted">France</span>
      <span class="stoken st10 corrected">est</span>
    </div>
  </div>
  <p class="spec-note" id="specnote">4 tokens accepted, 1 corrected. 5 tokens generated in 1 large-model pass.</p>
</div>

<style>
.spec-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.spec-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.spec-label {
  font-family: monospace;
  font-size: 12px;
  color: #666;
  width: 90px;
  flex-shrink: 0;
}
.spec-tokens {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.stoken {
  padding: 4px 10px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 13px;
  font-weight: 600;
  opacity: 0;
  animation: fadeUp 0.25s forwards;
}
.accepted { background: #1a3b2a; border: 1px solid #2a6b4a; color: #a8fac4; }
.rejected  { background: #3b1a1a; border: 1px solid #6b2a2a; color: #faa8a8; }
.corrected { background: #3b2a1a; border: 1px solid #6b4a2a; color: #fac4a8; }
.st1  { animation-delay: 0.2s; }
.st2  { animation-delay: 0.5s; }
.st3  { animation-delay: 0.8s; }
.st4  { animation-delay: 1.1s; }
.st5  { animation-delay: 1.4s; }
.st6  { animation-delay: 1.8s; }
.st7  { animation-delay: 2.0s; }
.st8  { animation-delay: 2.2s; }
.st9  { animation-delay: 2.4s; }
.st10 { animation-delay: 2.6s; }
.spec-note {
  color: #666;
  font-size: 12px;
  margin: 14px 0 0 0;
  opacity: 0;
  animation: fadeUp 0.3s 3.0s forwards;
}
</style>

### flash attention

Standard attention builds the full `seq x seq` attention matrix in slow HBM (GPU main memory). For a 4096-token sequence with 128-dim heads: 4096 x 4096 = 128MB per head per layer, written and re-read multiple times.

Flash Attention tiles the computation. Breaks Q, K, V into blocks, computes attention in tiles that fit in fast SRAM, keeps a running softmax and weighted sum, never writes the full matrix to HBM. Same mathematical output, far less memory traffic.

<div class="flash-viz">
  <p class="anim-label">Standard attention vs Flash Attention: where the matrix lives</p>
  <div class="flash-compare">
    <div class="flash-side">
      <p class="flash-side-lbl">Standard attention</p>
      <div class="flash-grid" id="flash-std-grid"></div>
      <p class="flash-mem-lbl hmem-lbl">HBM (slow DRAM) &mdash; O(N&sup2;) memory</p>
    </div>
    <div class="flash-side">
      <p class="flash-side-lbl">Flash Attention</p>
      <div class="flash-grid" id="flash-fa-grid"></div>
      <p class="flash-mem-lbl smem-lbl">SRAM (on-chip, fast) &mdash; O(N) memory</p>
    </div>
  </div>
  <p class="flash-note">Left: full NxN matrix written to slow HBM every layer. Right: 2x2 tiles computed one at a time in fast SRAM, result accumulated, matrix never materialized.</p>
</div>

<script>
window._anims = window._anims || {};
(function(){
  var flashTimers=[];
  window._anims.flash = function(){
    flashTimers.forEach(clearTimeout); flashTimers=[];
    var N=4;
    var stdG=document.getElementById("flash-std-grid");
    var faG=document.getElementById("flash-fa-grid");
    if(!stdG||!faG) return;
    stdG.innerHTML=''; faG.innerHTML='';
    var stdCells=[], faCells=[];
    for(var r=0;r<N;r++){
      for(var c=0;c<N;c++){
        var s=document.createElement("div"); s.className="fcell"; stdG.appendChild(s); stdCells.push(s);
        var f=document.createElement("div"); f.className="fcell"; faG.appendChild(f); faCells.push(f);
      }
    }
    flashTimers.push(setTimeout(function(){
      stdCells.forEach(function(c){ c.className="fcell fcell-hbm"; });
    },400));
    var tileSize=2, t=0, tileDelay=500;
    for(var tr=0;tr<N;tr+=tileSize){
      for(var tc=0;tc<N;tc+=tileSize){
        (function(row,col,d){
          flashTimers.push(setTimeout(function(){
            var idxs=[];
            for(var dr=0;dr<tileSize;dr++) for(var dc=0;dc<tileSize;dc++) idxs.push((row+dr)*N+(col+dc));
            idxs.forEach(function(i){ faCells[i].className="fcell fcell-sram"; });
            setTimeout(function(){ idxs.forEach(function(i){ faCells[i].className="fcell fcell-done"; }); },350);
          },d));
        })(tr,tc,t*tileDelay+700);
        t++;
      }
    }
  };
})();
</script>

<style>
.flash-viz {
  background: #0f1923;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 20px 0;
}
.flash-compare {
  display: flex;
  gap: 32px;
  justify-content: center;
  margin: 14px 0 8px;
}
.flash-side { flex: 1; max-width: 180px; }
.flash-side-lbl {
  font-family: monospace;
  font-size: 12px;
  color: #888;
  margin: 0 0 8px;
  text-align: center;
}
.flash-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 3px;
}
.fcell {
  aspect-ratio: 1;
  border-radius: 3px;
  background: #111820;
  border: 1px solid #1a2230;
  transition: background 0.3s, border-color 0.3s, box-shadow 0.3s;
}
.fcell-hbm { background: #3b1a1a; border-color: #6b2a2a; }
.fcell-sram { background: #1a3b2a; border-color: #2a6b4a; box-shadow: 0 0 5px #2a6b4a80; }
.fcell-done { background: #1a2a1a; border-color: #2a3b2a; }
.flash-mem-lbl {
  font-family: monospace;
  font-size: 11px;
  margin: 8px 0 0;
  text-align: center;
}
.hmem-lbl { color: #8a4a4a; }
.smem-lbl { color: #4a8a4a; }
.flash-note {
  color: #666;
  font-size: 12px;
  margin: 10px 0 0;
}
</style>

Benefits: O(seq) memory instead of O(seq^2), better GPU memory bandwidth utilization. For 4k-16k token sequences, Flash Attention 2 runs 2-4x faster than standard attention. Flash Attention 3 pushes further on H100s with async pipeline execution across warp groups.

**Sparse/Structured Attention** for very long contexts: full attention at 128k tokens is prohibitive even with Flash Attention. Options:
- Sliding window: each position attends only to the last w positions. O(seq) compute instead of O(seq^2).
- Block-sparse: some layers use local windows, others use global tokens. Used in Qwen2.5 for 128k contexts.

```python
# Using Flash Attention in Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

![Inference optimization overview](/assets/images/chatgpt/llminference.png){: loading="lazy"}

![Memory movements in GPU inference](/assets/images/chatgpt/memory_movements.png){: loading="lazy"}

---

## engineering tricks at scale

### prompt caching

The system prompt ("You are a helpful assistant...") is identical for every request in a deployment. Computing its KV cache from scratch every time wastes compute.

Cache the KV states for the system prompt once, reuse across all requests:

```text
Without prompt caching:
  Request 1: compute KV for [system(400 tokens) + user(100 tokens)] -> 500 tokens prefill
  Request 2: compute KV for [system(400 tokens) + user(80 tokens)]  -> 480 tokens prefill

With prompt caching:
  Startup:   compute KV for [system(400 tokens)]  -> cached
  Request 1: load cached + compute [user(100 tokens)] -> 100 tokens prefill (5x less)
  Request 2: load cached + compute [user(80 tokens)]  -> 80 tokens prefill
```

OpenAI charges 50% less for cached input tokens. Anthropic caches for 5 minutes. Saves both cost and latency since prefill is compute-bound. Free money if your system prompt is long.

### model routing

Not every question needs GPT-4. "What's 2+2?" doesn't need the same compute as "Write a distributed cache in Rust."

A lightweight classifier (adds ~2ms) routes requests to the cheapest model that can handle them:

```text
Query                   -> Model             -> Cost
"Hi"                    -> GPT-3.5-turbo     -> ~$0.001
"Summarize this email"  -> GPT-4o-mini       -> ~$0.01
"Debug this kernel"     -> GPT-4o            -> ~$0.05
"Prove this theorem"    -> o3                -> ~$0.20
```

Companies report 40-60% cost reduction from routing alone.

### semantic caching

If 1000 users ask "What is the capital of France?" within an hour, there's no reason to run inference 1000 times.

Embed user queries. Check cosine similarity against a cache. If similarity > 0.95, return the cached response directly.

```text
User: "What's the capital of France?"
  -> embed query
  -> cosine similarity search against cache
  -> match found (similarity=0.97) with "What is France's capital?"
  -> return cached response (0ms inference)
```

Breaks for creative or personalized queries: two users asking for a poem should get different answers, not a cache hit.

### prefix sharing (RadixAttention)

In multi-turn conversation, every new message re-sends the entire conversation history. The KV cache for previous turns was already computed.

SGLang's RadixAttention stores KV caches in a radix tree indexed by token prefix. New requests sharing a prefix with an old one reuse the cached states:

```text
Turn 1: [system + user_1]                                -> compute KV, cache
Turn 2: [system + user_1 + assistant_1 + user_2]         -> reuse first chunk
Turn 3: [system + user_1 + ... + assistant_2 + user_3]   -> more reuse
```

In long multi-turn conversations, this skips 80-90% of prefill computation.

### disaggregated serving

Prefill and decode have different hardware requirements:
- Prefill is compute-bound (lots of matrix math, GPU cores matter)
- Decode is memory-bandwidth-bound (loading model weights for each token)

Run them on separate GPU pools:

```text
Prefill Pool (high-compute H100 SXM):
  Process all input tokens in parallel
  Transfer KV cache to decode pool
        |
        v
Decode Pool (bandwidth-optimized):
  Generate tokens one-by-one using cached KV
```

Neither pool gets bottlenecked by the other's workload. Each scales independently.

### chunked prefill

For very long inputs (32k+ tokens), prefill can take seconds and blocks the entire GPU.

Break the input into chunks (e.g., 2048 tokens each) and interleave prefill chunks with decode steps from other requests. Prevents long prompts from creating latency spikes for everyone else in the batch.

### graceful degradation under load

When traffic spikes (viral tweet about ChatGPT), the system can't crash:

```text
Load Level    Action
Normal        GPT-4o, full context, streaming
High          Route new requests to GPT-4o-mini
Very High     Reduce max_tokens from 4096 to 1024
Extreme       Queue with estimated wait time
Critical      "We're experiencing high demand"
```

Priority queuing: paying API customers get priority over free-tier ChatGPT users.

### quick reference

| Problem | Trick | Savings |
|---------|-------|---------|
| Redundant system prompt computation | Prompt caching | 50% input cost |
| Slow decode (sequential bottleneck) | Speculative decoding | 2-3x speedup |
| Expensive model for easy questions | Model routing | 40-60% cost |
| Same question asked 1000x | Semantic caching | ~100% for cache hits |
| Redundant work in multi-turn chat | Prefix sharing | 80-90% prefill savings |
| Prefill and decode fight for GPU | Disaggregated serving | Better utilization |
| Long prompts block other requests | Chunked prefill | Lower P99 latency |
| Traffic spikes | Graceful degradation | Availability |

---

## evaluation

Perplexity (cross-entropy loss on held-out text) tracks how well a model predicts unseen text. Useful for comparing checkpoints during training. Useless for predicting whether the model gives good answers.

The standard benchmarks - MMLU (57-subject multiple choice), HumanEval (164 Python problems run against test cases), GSM8K (math word problems) - share the same failure mode: once a benchmark matters, labs optimize for it. Training data accumulates benchmark-adjacent content. Checkpoints get selected by score. GPT-4 hit 86.4% on MMLU at launch; models trained on GPT-4's public outputs score higher. The number goes up without the capability following.

Chatbot Arena avoids this by having real users compare two anonymous models on their actual tasks. No fixed test set. ELO from millions of blind comparisons. When Arena rankings disagree with benchmark rankings, Arena is usually right.

Human evaluation (annotators rating outputs on helpfulness rubrics) is what RLHF trains on. It's also slow, expensive, and systematically biased: annotators prefer longer and more confident answers, so models learn to sound helpful. For open-ended questions there's no ground truth, and that bias compounds across training.

---

## the full picture

From a URL in a web crawl to a token in your browser: crawl the web, clean it, tokenize it, train a transformer to predict next tokens, fine-tune it to follow instructions, serve it through a layered production system, and stream each token back as it's generated.

Pretraining costs $50-100M+ and runs for months on thousands of GPUs. SFT and RLHF run for a few weeks and cost a few million at most. The ongoing engineering challenge is inference: routing, batching, caching, safety, streaming, and observability for millions of concurrent users.

Next-token prediction on 13 trillion tokens is a simple objective. The weights that result encode facts, reasoning patterns, and enough of a world model to pass the bar exam. The model generates the most probable continuation regardless of truth. When training data is sparse on a topic, plausible text and true text look identical to the predictor - so it confabulates.

---

## further reading

**Papers**: [InstructGPT](https://arxiv.org/abs/2203.02155) (RLHF in practice), [LLaMA 3](https://arxiv.org/abs/2407.21783) (training at scale), [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) (inference at scale), [FlashAttention-2](https://arxiv.org/abs/2307.08691) (memory-efficient attention)

**Articles**: [Quantization explained](https://ngrok.com/blog/quantization) (ngrok - INT4/INT8 quantization in practice)

**Hands-on**: [Karpathy - Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxruWIwMSOWxAywIRyoiOWHaXY), [makemore](https://github.com/karpathy/makemore), [Hugging Face NLP course](https://huggingface.co/learn/nlp-course)

---

## summary

Pretraining predicts the next token on trillions of tokens of internet text. The supervision signal is the text itself. The weights that emerge encode language, facts, and reasoning patterns, because those are what make text predictable. Models hallucinate for the same reason: they generate the most plausible continuation, and plausible is not the same as true.

Post-training shapes how the model uses what it learned. SFT teaches response format using (instruction, response) pairs. RLHF uses a reward model trained on human preference rankings to steer behavior via PPO. DPO does the same thing without the reward model or RL loop - directly maximizing the likelihood of preferred responses relative to rejected ones.

Attention computes dot products between query and key vectors, applies softmax to get weights, then takes a weighted sum of value vectors. Every token attends to every previous token. Multi-head attention runs this in parallel with different projections, letting different heads specialize on syntax, coreference, and other relationship types.

The KV cache saves the key and value vectors for all previous tokens so decode doesn't recompute them each step. Without it, decode is O(n^2) in sequence length. With it, O(n) per step. GQA groups heads to share K/V matrices, shrinking the cache by 4-8x with minimal quality loss.

Quantization stores weights in INT8 or INT4 instead of FP32. A 7B model in INT4 fits in 3.5GB instead of 14GB. PTQ rounds after training; GPTQ/AWQ calibrate per-layer to minimize output error. QLoRA fine-tunes quantized models by training low-rank adapters in higher precision while keeping the base frozen.

Flash Attention tiles the Q/K/V computation to stay in fast on-chip SRAM rather than writing the full NxN attention matrix to slow HBM. Mathematically identical output. O(N) memory instead of O(N^2). 2-4x faster for sequences above 4k tokens.

Speculative decoding runs a small draft model to generate candidate tokens, then verifies all of them with the large model in a single forward pass via rejection sampling. 2-3x speedup with no quality change, because text is predictable enough that the small model gets most tokens right.

TTFT (Time to First Token) and TPS (Tokens Per Second) are the two metrics that determine perceived quality. Prefill time and queue wait drive TTFT. Model size and GPU memory bandwidth drive TPS. Streaming hides most of the total latency by delivering the first token in ~100-200ms while generation continues in the background.

<style>
.anim-replay-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  background: #111820;
  border: 1px solid #2a4a6b;
  color: #6a9acb;
  font-family: monospace;
  font-size: 12px;
  cursor: pointer;
  padding: 3px 10px;
  border-radius: 4px;
  opacity: 0.6;
  transition: opacity 0.2s, color 0.2s;
  line-height: 1.5;
  z-index: 10;
}
.anim-replay-btn:hover { opacity: 1; color: #a8c7fa; border-color: #4a7acb; }
.token-animation, .attn-viz, .rlhf-loop, .topp-viz, .pipeline-viz, .spec-viz,
.cbatch-viz, .autogen-viz, .ctx-win-viz, .moe-viz, .flash-viz, .kv-viz {
  position: relative;
}
.token { transition: transform 0.15s; }
.token:hover { transform: scale(1.1); cursor: default; }
</style>

<script>
(function(){
  window._anims = window._anims || {};

  window._anims.kvcache = function(){
    var c=document.getElementById("dtokens");
    var n=document.getElementById("kvnote");
    if(!c) return;
    c.innerHTML='';
    if(n){ n.textContent=''; n.style.opacity='0'; }
    ["The"," capital"," of"," France"," is"," Paris","."].forEach(function(t,i){
      setTimeout(function(){
        var el=document.createElement("span");
        el.className="kv-token"; el.textContent=t; el.style.animationDelay="0s";
        c.appendChild(el);
        if(i===6) setTimeout(function(){ if(n){ n.textContent="Cache grows with each new token. Longer context = more VRAM."; n.style.opacity="1"; } },400);
      },800*(i+1));
    });
  };

  function resetCSS(container){
    var items=[];
    container.querySelectorAll('*').forEach(function(el){
      var a=window.getComputedStyle(el).animationName;
      if(a && a!=='none'){ items.push(el); el.style.animation='none'; }
    });
    void container.offsetWidth;
    items.forEach(function(el){ el.style.animation=''; });
  }

  var fnMap={
    'autogen-viz': function(){ if(window._anims.autogen) window._anims.autogen(); },
    'ctx-win-viz': function(){ if(window._anims.ctxwin)  window._anims.ctxwin();  },
    'moe-viz':     function(){ if(window._anims.moe)     window._anims.moe();     },
    'flash-viz':   function(){ if(window._anims.flash)   window._anims.flash();   },
    'kv-viz':      function(){ if(window._anims.kvcache) window._anims.kvcache(); },
  };

  function getReplayFn(el){
    for(var k in fnMap){ if(el.classList.contains(k)) return fnMap[k]; }
    return null;
  }

  function addReplayBtn(container, fn){
    if(container.querySelector('.anim-replay-btn')) return;
    var btn=document.createElement('button');
    btn.className='anim-replay-btn';
    btn.textContent='↺ replay';
    btn.addEventListener('click',function(e){
      e.stopPropagation();
      if(typeof fn==='function') fn();
      else resetCSS(container);
    });
    container.appendChild(btn);
  }

  function init(){
    var sel='.token-animation,.attn-viz,.rlhf-loop,.topp-viz,.pipeline-viz,.spec-viz,.cbatch-viz,.autogen-viz,.ctx-win-viz,.moe-viz,.flash-viz,.kv-viz';
    var containers=document.querySelectorAll(sel);

    if('IntersectionObserver' in window){
      var triggered=new Set();
      var obs=new IntersectionObserver(function(entries){
        entries.forEach(function(entry){
          var c=entry.target;
          if(entry.isIntersecting && !triggered.has(c)){
            triggered.add(c);
            var fn=getReplayFn(c);
            if(fn) fn(); else resetCSS(c);
          }
        });
      },{threshold:0.25,rootMargin:'0px 0px -40px 0px'});
      containers.forEach(function(c){ obs.observe(c); });
    } else {
      containers.forEach(function(c){ var fn=getReplayFn(c); if(fn) fn(); });
    }

    containers.forEach(function(c){
      addReplayBtn(c, getReplayFn(c) || function(){ resetCSS(c); });
    });
  }

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded',init);
  } else {
    init();
  }
})();
</script>
