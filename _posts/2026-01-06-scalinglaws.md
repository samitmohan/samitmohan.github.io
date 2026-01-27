---
layout: post
title:  "scaling laws"
date:   2026-01-06 14:06:04 +0530
categories: tech
---

## Question

What can LLMs do that classical ML can't? At scale, how do LLM's differ from traditional ML? What does this mean for the future of AI? Does more data mean better results? OpenAI wrote a paper just on this, it's called [**Scaling Laws for Neural Language Models**](https://arxiv.org/pdf/2001.08361) in 2020.

-----

## Scaling Laws

![sl](/assets/images/scaling_laws/scaling_laws.png)

SL: simple predictive laws / rules for behaviour of language model performance.

*HOW WE TEST::*

- OLD: Tune hyper-parameters on big models
- NEW: Tune on small models -> extrapolate to large ones

Scaling Laws Paper says validation and test loss decreases as parameter and number of layers increases along with compute increase.

**IDEA:** Do all experimentatioons on small model with less compute -> Nail the big model in one-go.

> Scaling laws are very natural thing to think about for data -> as we increase size of data/model -> we expect certain behaviour out of the model.

*Questions we should ask ourselves:*

- Data vs performance (Are there simple rules that determine how data affects performance?)
- Data vs model size (Do we train on more data or bigger models)
- Hyper parameters vs performance (How should we set hyperparameter on big models)

**Maybe intelligence -> lot of compute applied to lot of data having lot of parameters.**

This was the first idea of scaling laws in the 1970s^

- We also need to train on enough data which is very important (GPT3 was undertrained) 
- Chinchilla (half parameter size of GPT3 (70b) but 4x data -> performed better) 

Data Scaling Laws: formula that maps dataset size(n)

Loss and 'n' is linear on a loglog plot

![dl](/assets/images/scaling_laws/data_loss.png)

**Engineering Data Laws:**

How does data composition affect model performance (not just size) -> data composition only affects offset not slope (You can do data select experiments on a much smaller model)

![datacomposition](/assets/images/scaling_laws/dc.png)

Another question to look for:

- We have finite data, how does repeating examples affect scaling?
        - Upto 4 epochs repeating data is almost as good as new but after that it rapidly diminishing returns.
- Given that repeated data is less valuable -> data selection should adapt to scale.
- **Repeat high quality data OR include new data (trade-off)**

### How can we design a huge LM? What to pick?

- **Architecture:** LSTM vs Transformer (Transformer loss decreases as we increase parameters (MOE is the only thing better than vanilla transformer))
- **Optimiser:** Adam is much better than SGD as we increase epochs (adaptive learning rate (basically takes steps automatically instead of a fixed size))
- **Depth:** layers >=6 is good, 1 vs 2 layers make huge difference after that we have plateud.
- **Batch Size:** batch size increase -> gradient steps increase (past certain point -> diminishing returns (bias dominates instead of learning deeper features)
  - Critical BatchSize: minimum number of steps for target loss (compute increase -> steps can stay the same (BS fixed))

### Side note

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

*Why parameters alone are not enough?* : We need enough training (on enough data with good compute)

If:

- Model is huge
- Data is small

Then:

- Model memorizes
- Poor generalization

Hence: Big models need big data.

> Compute ∝ Parameters ∝ Training ∝ Tokens

### Recap

- Scaling laws describe how the performance of large language models (LLMs) improves predictably as you increase certain factors. These factors are the size of the model (measured in parameters), the size of the training dataset, and the amount of compute used for training
- Increase the number of parameters in a model without also scaling the dataset or compute, you’ll hit diminishing returns. The same goes for the other factors. Scaling one without the others doesn’t work
- Shift to GPUs was a breakthrough, allowing researchers to scale up both model size and dataset size. Transformers have made this even better.

> This is CAP Theorem but for LLMs

For a fixed data distribution and objective, expected loss improves predictably as you scale:

- Compute
- Size
- Parameters

You cannot simultaneously maximize:

- Model size (parameters)
- Training data
- Compute efficiency (time & cost)

If you push two aggressively, the third suffers.

> Loss ∼ 𝑓 ( parameters , data , compute)

Case 1: Huge model + limited compute

- Undertrained model
- Poor generalization
- Wasted parameters

Case 2: Huge data + limited parameters

- Underfitting
- Can’t absorb structure

Case 3: Huge model + huge data

- Requires massive compute
- Training becomes infeasible

Just like CAP: You must choose your tradeoff point.

**GPT vs Deepseek:**
ChatGPT chooses:

- Large parameters
- Broad data
- Massive compute ➡

>️ Optimizes generality and robustness

DeepSeek chooses:

- Fewer parameters
- Higher-quality data
- More inference-time compute

> Optimizes reasoning efficiency

They’re on different vertices of the triangle, just like different databases choose CP or AP.
CAP theorem (hard constraint):

```text
        Consistency
            ▲
            |
            |
Availability ───── Partition tolerance


LLM scaling (soft frontier):

          Model Quality
               ▲
               |
               |
Compute  ───── Data
```

CAP says you must choose what to give up; scaling laws say you must choose what to pay for.

More difference between GPT and Deepseek:

```text
ChatGPT:  More parameters + less per-query thinking
DeepSeek: Fewer parameters + more per-query thinking
```

> Both obey scaling laws — just along different axes.

Scaling laws apply differently to DeepSeek and ChatGPT because they optimize different objectives over different data distributions. ChatGPT follows classic scaling laws by increasing parameters, data, and compute to reduce expected loss across a broad range of tasks, yielding robust general intelligence. DeepSeek, by contrast, specializes in reasoning-heavy tasks and achieves higher parameter efficiency through curated data, teacher-student distillation, and increased inference-time computation. This shifts DeepSeek onto a steeper but narrower scaling curve, allowing strong performance with fewer parameters without violating scaling laws.

### Limitations of Scaling Laws + Future

But we are going out of data:

- Running out of data (quality of data) -> ways to make data synthetic (deepmind alphago plays against itself and only has synthetic data) 
- Reasoning model (o1) bridges this gap (chain of thought) -> longer o1 thinks better it performs (new paradigm for scaling llms -> reasoning (we need higher compute for this)- > current state of AI)
- Invent new arch? Numerical stability of model should be there (none so far, transformer works only)
![sl](/assets/images/scaling_laws/scaling_loss.png)