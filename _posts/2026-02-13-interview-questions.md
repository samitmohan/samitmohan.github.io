---
layout: post
title: "favourite interview questions"
date: 2026-02-13 14:06:04 +0530
categories: tech
tokens: "~6.5k"
description: "Five questions I'd rather ask than 'derive Kadane's algorithm in 30 minutes pretending you've never seen it.'"
---

These are some of my favourite questions to ask in interviews and I wish we move forward and have these questions in interview rounds instead of cramming algorithmic questions and deriving kadane's algorithm in 30 minutes, pretending we have never seen it before.

---

1. **DSA + Hardware**: Matrix multiplication and how to optimise it
2. **Math + Probability**: Estimating Pi with Monte Carlo (JomaTech)
3. **Networking**: What happens when you type google.com
4. **GPU + OS**: CUDA and Flash Attention
5. **Modern ML**: Implement a GPT from scratch (autograd, attention, the works)

---

## Q: What is the complexity of matrix multiplication? Write it. How can you optimise it?

> O(n^3)

First you check if they can even be multiplied. Then you create an output array to store the results. Then you run a triple loop.

```python

import numpy as np
def matrixmul(a, b):
    m, n1 = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    if n1 != n2: return -1
    # else do matrix mul
    c = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n1):
                c[i][j] += a[i][k] * b[k][j]
    return c

def matrixmulNumpy(a, b):
    a, b = np.array(a), np.array(b)
    if a.shape[1] != b.shape[0]: return -1
    return (a @ b).tolist()
    # return np.matmul(A,B).tolist()
```

How do you optimise it? Three levels.

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/interview/cuda_parallel_poster.jpg">
  <source src="/assets/images/interview/cuda_parallel.mp4" type="video/mp4">
</video>
*CPU processes matrix elements one at a time. GPU processes them all at once.*

**Strassen's algorithm** - Instead of 8 recursive multiplications on sub-matrices, Strassen does it in 7. Drops O(n^3) to O(n^2.807). The constant factor is huge though, so in practice you switch to Strassen for large matrices and fall back to naive for small ones.

**Cache-aware blocking** - Modern CPUs have cache hierarchies (L1, L2, L3). The naive i-j-k loop gets cache misses on every access to `b` because you're jumping across rows. Tile the multiplication into blocks that fit in L1 cache. Doesn't change big-O but the wall-clock difference is massive.

**Let Fortran do it** - When you write `a @ b` in numpy, you're not running Python. You're calling BLAS (Basic Linear Algebra Subprograms) - Fortran routines hand-optimised for decades with SIMD instructions, cache blocking, and loop unrolling. This is why numpy is fast. Python is just the steering wheel.

```python
import numpy as np
import time

n = 500
a, b = np.random.rand(n, n), np.random.rand(n, n)

start = time.time()
c = a @ b
print(f"numpy (BLAS) {n}x{n}: {time.time() - start:.4f}s")

# now try the naive python triple loop with n=500 and see how long do you have to wait...
```

---

## Q: Given a function that generates a random number between 0 and 1 (uniformly distributed), calculate Pi.

> first response when I ask this is "what??"

Draw a unit square. Now draw a quarter circle of radius 1 inside it, center at the origin.

Area of the square = 1. Area of the quarter circle = Pi * r^2 / 4 = Pi/4.

Now throw random darts at the square. Each dart lands at a random point (x, y) where both x and y come from your uniform random function. If x^2 + y^2 <= 1, the dart landed inside the quarter circle. The fraction of darts inside converges to the ratio of areas: Pi/4.

So Pi = 4 * (inside / total). That's it. Monte Carlo simulation - using randomness to solve a deterministic problem.

```python
import random

def estimate_pi(n):
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / n

# more darts = better estimate
for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    print(f"n={n:>10,} -> pi = {estimate_pi(n):.6f}")
```

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/interview/monte_carlo_poster.jpg">
  <source src="/assets/images/interview/monte_carlo.mp4" type="video/mp4">
</video>
*200 random darts, Pi estimate converging toward 3.14159...*

Where I found this problem -> [Joma Tech](https://www.youtube.com/watch?v=pvimAM_SLic&t=68s)

---

## Q: What happens when you enter www.google.com and press enter?

> This was inspired by this [repo](https://github.com/alex/what-happens-when) I read long time ago

**DNS Resolution** - Browser needs an IP address. "www.google.com" means nothing to the network. It checks: browser cache -> OS cache -> router cache. If all miss: recursive DNS resolver asks root nameserver -> ".com" TLD nameserver -> Google's authoritative nameserver. You get back something like 142.250.80.4.

**TCP Handshake** - Three packets to establish a reliable connection. SYN, SYN-ACK, ACK. Both sides agree they can talk before any data flows. This is why TCP is "reliable" - there's a contract.

**TLS Handshake** - Because HTTPS. Client sends supported cipher suites, server sends its certificate (signed by a CA), they do a Diffie-Hellman key exchange to agree on a shared secret. Everything after this is encrypted with symmetric keys derived from that secret.

**HTTP Request** - `GET / HTTP/2` with headers: cookies, user-agent, accept-encoding, etc.

**Server Side** - Google's load balancer routes your request to one of thousands of servers. Server generates the HTML response for the search homepage.

**Browser Rendering** - Parse HTML into a DOM tree. Parse CSS into CSSOM. Merge them into a render tree. Compute layout (geometry of every element). Paint pixels. Composite layers for GPU acceleration.

Under 200ms for the whole thing on a good connection. Most of that is network latency, not computation.

You can see every step in code:

```python
import socket
import ssl

# step 1: DNS
ip = socket.getaddrinfo('www.google.com', 443)[0][4][0]
print(f"DNS: www.google.com -> {ip}")

# step 2: TCP + TLS
sock = socket.create_connection(('www.google.com', 443))
context = ssl.create_default_context()
ssock = context.wrap_socket(sock, server_hostname='www.google.com')
print(f"TLS: {ssock.version()}")

# step 3: raw HTTP request
ssock.sendall(b'GET / HTTP/1.1\r\nHost: www.google.com\r\nConnection: close\r\n\r\n')
response = ssock.recv(512).decode()
print(response[:200])
ssock.close()
```

Three stdlib imports and you've just done DNS resolution, a TCP handshake, a TLS handshake, and an HTTP request. Everything your browser does, in 10 lines.

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/interview/network_stack_poster.jpg">
  <source src="/assets/images/interview/network_stack.mp4" type="video/mp4">
</video>
*DNS, TCP handshake, TLS, HTTP request/response - the full flow.*

I like explanations that start from first principles at the hardware level, what happens from the moment you press Enter: the keyboard controller fires a hardware interrupt, the CPU jumps to the interrupt handler, the OS processes the keycode, and so on up the stack

---

## Q: How does CUDA work? What is Flash Attention?

> a must for any deep learning interview, do you just call torch.device("cuda") or do you know what happens

CUDA is NVIDIA's model for programming GPUs. A GPU has thousands of cores - each one dumber than a CPU core, but there are thousands of them running in parallel. CUDA lets you write "kernels": functions that execute across all those cores simultaneously. When PyTorch does `torch.matmul(A, B)`, a CUDA kernel tiles the matrices, loads chunks into fast SRAM, computes, writes back to global memory. That's the gap between MicroGPT (Python for loops) and production (thousands of parallel threads).

Flash Attention is the insight that standard attention is memory-bound, not compute-bound. Naive attention computes the full N x N attention matrix, writes it to slow HBM, reads it back. Flash Attention never materializes the full matrix. It processes attention in tiles - loading chunks of Q, K, V into SRAM and using the online softmax trick (keeping running max and sum statistics) to compute exact attention without the N x N memory hit. Same math. Way less memory. Faster in practice because you avoid the memory bottleneck entirely.

The difference between CPU and GPU in one example:

```python
# CPU: one thread loops over everything sequentially
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
```

```c
// GPU (CUDA): thousands of threads, each computes ONE element
__global__ void matmul(float* A, float* B, float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[i*N + k] * B[k*N + j];
    C[i*N + j] = sum;
}
// launched with <<<grid, block>>> - one thread per output element
```

Same triple loop. The CPU does N^2 iterations of the outer two loops sequentially. The GPU launches N^2 threads and does them all at once. That's the entire idea.

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/interview/flash_attention_poster.jpg">
  <source src="/assets/images/interview/flash_attention.mp4" type="video/mp4">
</video>
*Standard attention materializes the full NxN matrix in slow HBM. Flash Attention tiles through fast SRAM.*

---

## Q: Implement a GPT from scratch. No libraries.

> ~200 lines. No `nn.Module`, no `loss.backward()`, no CUDA kernels.

I put together a [visual walkthrough](/assets/images/interview/microgpt_config.png) of MicroGPT. The diagrams below are from that document.

<img src="/assets/images/interview/microgpt_architecture.png" alt="MicroGPT GPT Architecture - decoder-only transformer pipeline from token input to next-token prediction" style="max-width:100%">
*The `gpt()` function as a pipeline: embedding lookup, RMSNorm, multi-head attention with residual, MLP with residual, lm_head projection, softmax.*

### Autograd: Why It Matters

Autograd - automatic differentiation - is the engine that makes neural networks trainable. The core idea: every mathematical operation you perform during the forward pass gets recorded in a computational graph. When you're done and you have a loss value, you walk that graph backwards to compute how much each parameter contributed to the error. That's backpropagation, and it's nothing more than the chain rule applied systematically.

In PyTorch, this is hidden behind `loss.backward()`. You never see the graph. You never see the gradients flowing. MicroGPT rips that abstraction away by implementing autograd at the scalar level.

### The Value Class: Building the Computational Graph

The `Value` class is the foundation of everything. Each `Value` wraps a single floating-point number and tracks three things:

1. **`data`** - the actual number (used in the forward pass)
2. **`grad`** - the derivative of the final loss with respect to this number (filled in during the backward pass)
3. **`_children` and `local_grads`** - pointers to the inputs that created this value, along with the local derivative of this operation with respect to each input

Every time you do `a + b` or `a * b` where `a` and `b` are `Value` objects, a new `Value` is created that remembers its parents and the local gradients. Addition: the local gradient with respect to both inputs is `1.0` (because d(a+b)/da = 1 and d(a+b)/db = 1). Multiplication: the local gradient with respect to `a` is `b.data` and vice versa (because d(a*b)/da = b).

This builds a DAG - a directed acyclic graph - where every node is a scalar and every edge represents a mathematical dependency.

| Feature | MicroGPT (`Value`) | PyTorch (`torch.Tensor`) |
| :--- | :--- | :--- |
| **Granularity** | **Scalar-based**. Every single number is an object. | **Tensor-based**. Operates on n-dimensional arrays. |
| **Backprop** | Manual topological sort and chain rule. | Highly optimized C++/CUDA kernels for DAG traversal. |
| **Visibility** | You can see every gradient flow through every node. | Hidden behind the `loss.backward()` black box. |

In PyTorch, we write `x = torch.randn(10, 10, requires_grad=True)`. In MicroGPT, we initialize a list of 100 individual `Value` objects. This makes the chain rule visceral: calling `backward()` literally traverses the history of every addition and multiplication.

### The Chain Rule and DAG Traversal

Here is the key insight. If you have a chain of operations like `loss = f(g(h(w)))`, then:

```
d(loss)/d(w) = d(loss)/d(f) * d(f)/d(g) * d(g)/d(h) * d(h)/d(w)
```

Each factor in that product is a "local gradient" - the derivative of one operation with respect to its immediate input. The chain rule says: to get the gradient of the loss with respect to any parameter, multiply all the local gradients along the path from that parameter to the loss.

But in a real network, the graph is not a simple chain. It's a DAG with fan-out (one value feeds into multiple operations) and fan-in (one operation takes multiple inputs). So `backward()` does two things:

1. **Topological sort** - it orders every node so that a node always comes after all the nodes that depend on it. This is done with a simple recursive DFS.
2. **Reverse traversal** - starting from the loss (whose gradient is seeded at 1.0, because d(loss)/d(loss) = 1), it walks backwards through the sorted list. For each node, it pushes `local_grad * node.grad` into each child's `.grad` field.

The `+=` is critical. When a value feeds into multiple downstream operations, its gradient accumulates contributions from all of them. This handles the fan-out case correctly: if `w` is used in both `a = w * 3` and `b = w * 5`, then `d(loss)/d(w) = d(loss)/d(a) * 3 + d(loss)/d(b) * 5`.

That's the entire backward pass. No magic. Just: sort the graph, walk it backwards, multiply and accumulate local gradients.

<img src="/assets/images/interview/microgpt_autograd.png" alt="Autograd Value class - scalar computation graph with backward() chain rule, showing L=(a*b)+c example with gradient flow" style="max-width:100%">
*Every operation builds a DAG node. backward() walks it in reverse, multiplying local gradients by the chain rule.*

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/interview/comp_graph_poster.jpg">
  <source src="/assets/images/interview/comp_graph.mp4" type="video/mp4">
</video>
*Forward pass builds the DAG, backward pass propagates gradients through it.*

#### What is a Gradient, Practically?

If you've forgotten your multivariable calculus: a gradient is just a "nudge." If a weight has a gradient of `0.5`, it means that if we increase that weight by a tiny amount, the loss will increase by half that amount. Our goal is to nudge every weight in the *opposite* direction of its gradient to minimize the loss.

### Wiring vs. Modules

MicroGPT doesn't use `nn.Module`. Instead, it uses raw Python list comprehensions to implement the math.

- **Linear Layers**: Instead of `nn.Linear(16, 16)`, MicroGPT uses:
  `[sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]`
- **Activation**: Instead of `nn.ReLU()`, it's a manual `max(0, x)` wrapper inside the `Value` class.
- **Normalisation**: `rmsnorm(x)` is a custom function calculating the root mean square of a list of scalars manually.

### Sequential Training and the KV Cache

This is the most significant departure from standard PyTorch training.

- **PyTorch Way**: We train on sequences in **parallel**. We pass a batch of `[Batch, Seq_Len]` and use a **Causal Mask** (a triangular matrix of $-\infty$) to hide future tokens.
- **MicroGPT Way**: It processes tokens **one by one** (`for pos_id in range(n)`). It appends keys and values to a list (a **KV Cache**) as it goes.

**Why?** Implementing a 2D causal mask and matrix multiplication using scalar `Value` objects would be catastrophically slow. By using a KV cache during training, Karpathy makes "causality" (not looking ahead) implicit and the code much easier to read.

<img src="/assets/images/interview/microgpt_attention.png" alt="Multi-head attention - Q,K,V projections split into 4 heads, scaled dot-product attention with KV cache" style="max-width:100%">
*4 heads x 4 dimensions = 16 total. Each head independently attends over the KV cache, then results are concatenated and projected through Wo.*

### Adam from Scratch

In PyTorch, we call `optimizer.step()`. MicroGPT manually tracks:

1. `m`: The first moment (momentum).
2. `v`: The second moment (uncentered variance).
3. `m_hat` / `v_hat`: Bias correction for the early steps of training.

It then updates `param.data` directly. This proves that Adam is just an adaptive learning rate that scales every single weight update based on its own history.

<img src="/assets/images/interview/microgpt_training.png" alt="Training loop - tokenize, forward pass, cross-entropy loss, backward pass, Adam optimizer update, zero gradients" style="max-width:100%">
*One complete training step: tokenize name, forward through gpt(), compute cross-entropy loss, backward(), Adam update with LR decay, zero gradients.*

### The Gradient Journey

When we call `loss.backward()`, we are executing a chain reaction of local derivatives.

Imagine the very last operation: `loss = total_loss / batch_size`.

- The `loss.grad` is seeded at `1.0`.
- This flows back to `total_loss.grad` as `1.0 / batch_size`.
- If `total_loss = loss1 + loss2`, then both `loss1.grad` and `loss2.grad` inherit `total_loss.grad` (because the derivative of `x+y` w.r.t `x` is `1`).

This continues until we reach the **Token Embeddings**. In MicroGPT, we can literally print `state_dict['wte'][10][5].grad` to see how much the 5th dimension of the 10th token's embedding contributed to the error of a specific name like "Andre-j".

### Numerical Stability: The `max_logit` Trick

In the `softmax` function, you'll see:
`max_logit = max(l.data for l in logits)`
`exp_logits = [(logit - max_logit).exp() for logit in logits]`

**Why?**
The exponential function `e^x` grows catastrophically fast. If a logit is `100`, `e^100` is ~2.6e43, which will overflow a floating-point number. By subtracting the maximum value, the largest value becomes `e^0 = 1`, and everything else becomes a small fraction between `0` and `1`. Since softmax is translation-invariant ($Softmax(x) = Softmax(x - c)$), the math remains identical, but the code stops crashing.

### Embedding + Positional Encoding

<img src="/assets/images/interview/microgpt_embedding.png" alt="Embedding and positional encoding - wte token lookup table and wpe position lookup table, element-wise addition" style="max-width:100%">
*wte (27x16) encodes what the token IS, wpe (16x16) encodes where the token SITS. Both are learnable - no sin/cos here.*

### The Code (200 Lines)

The code below is my own implementation, clocking in at just over 200 lines. It is designed to be read line-by-line, without the abstraction of modern deep learning frameworks. If you can walk through this code and explain every gradient, you know how a transformer works.

Check out Andrej's original blog [here](https://karpathy.github.io/2026/02/12/microgpt/) for the high-level context.

```python
import math
import random
import urllib.request

random.seed(42)

names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
urllib.request.urlretrieve(names_url, 'names.txt')

docs = [l.strip() for l in open('names.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)

print(f"Number of docs: {len(docs)}")

# Tokeniser
chars = sorted(set(''.join(docs))) # unique characters in the dataset
BOS = len(chars) # beginning of sequence token
vocab_size = len(chars) + 1 # number of unique characters + 1 for the BOS token

print(f"Vocab size: {vocab_size}")

# Autograd for DAG of operations

class Value:
    def __init__(self, data, _children=(), local_grads = ()):
        self.data = data # the value of this node (scalar (forward pass))
        self.grad = 0 # derivative of loss wrt this node
        self._children = _children # children of this node in the computational graph
        self.local_grads = local_grads # local derivatives of this node wrt its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other))
        out.local_grads = (1.0, 1.0)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other))
        out.local_grads = (other.data, self.data)
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self): return Value(math.log(self.data + 1e-10), (self,), (1/(self.data + 1e-10),))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __lt__(self, other): return self.data < (other.data if isinstance(other, Value) else other)
    def __gt__(self, other): return self.data > (other.data if isinstance(other, Value) else other)

    def backward(self):
        topological_graph = []
        visited = set()
        def build_top(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_top(child)
                topological_graph.append(v)

        build_top(self)
        self.grad = 1 # seed the gradient of the output node with 1.0 (dL/dL = 1)
        for v in reversed(topological_graph):
            for child, local_grad in zip(v._children, v.local_grads):
                child.grad += local_grad * v.grad


# Initialise the parameters of the model
n_embed = 16
n_head = 4
n_layer = 1
block_size = 16 # maximum context length for predictions
head_dim = n_embed // n_head # dimension of each attention head

# Randomly initialise the parameters of the model as Value objects
matrix = lambda rows, cols: [[Value(random.gauss(0, 0.08)) for _ in range(cols)] for _ in range(rows)]
state_dict = {'wte': matrix(vocab_size, n_embed), 'wpe': matrix(block_size, n_embed), 'lm_head': matrix(vocab_size, n_embed)}

for i in range(n_layer):
    state_dict[f'blocks.{i}.attn.wq'] = matrix(n_embed, n_embed)
    state_dict[f'blocks.{i}.attn.wk'] = matrix(n_embed, n_embed)
    state_dict[f'blocks.{i}.attn.wv'] = matrix(n_embed, n_embed)
    state_dict[f'blocks.{i}.attn.wo'] = matrix(n_embed, n_embed)
    state_dict[f'blocks.{i}.mlp_fc1'] = matrix(4 * n_embed, n_embed)
    state_dict[f'blocks.{i}.mlp_fc2'] = matrix(n_embed, 4 * n_embed)

params = [p for matrix in state_dict.values() for row in matrix for p in row] # flatten the parameters into a single list
print(f"Number of parameters: {len(params)}")

# model architecture -> GPT2

# Linear layer used for forward pass
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# Convert logits between 0-1
def softmax(logits):
    max_logit = max(l.data for l in logits)
    exp_logits = [(logit - max_logit).exp() for logit in logits] # subtract max logit for numerical stability
    sum_exp_logits = sum(exp_logits)
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

# Normalisation layer used in the transformer blocks (Linear layer + RMSNorm)
def rmsnorm(x):
    eps = 1e-8
    mean_square = sum(xi**2 for xi in x) / len(x)
    root_mean_square = (mean_square + eps)**0.5
    return [xi / root_mean_square for xi in x]

# Decoder-only transformer architecture
def gpt(token_id, pos_id, keys, values):
    token_embedding = state_dict['wte'][token_id] # get the token embedding for the input token ID
    position_embedding = state_dict['wpe'][pos_id] # get the position embedding for the input position ID
    x = [te + pe for te, pe in zip(token_embedding, position_embedding)] # add the token and position embeddings to get the input representation for the transformer block
    x = rmsnorm(x) # apply RMSNorm to the input representation
    for i in range(n_layer):
        # mha block
        x_residual = x # save the input to the MHA block for the residual connection
        x = rmsnorm(x)
        q = linear(x, state_dict[f'blocks.{i}.attn.wq']) # compute the query vectors for the MHA block
        k = linear(x, state_dict[f'blocks.{i}.attn.wk']) # compute the key vectors for the MHA block
        v = linear(x, state_dict[f'blocks.{i}.attn.wv']) # compute the value vectors for the MHA block
        keys[i].append(k)
        values[i].append(v)
        x_attention = []
        for h in range(n_head):
            hsize = h * head_dim # compute the start index of the current head in the query/key/value vectors
            qh = q[hsize:hsize+head_dim] # get the query vector for the current head
            kh = [ki[hsize:hsize+head_dim] for ki in keys[i]] # get the key vectors for the current head from all previous time steps
            vh = [vi[hsize:hsize+head_dim] for vi in values[i]] # get the value vectors for the current head from all previous time steps
            attention_scores = [sum(qhi * khi for qhi, khi in zip(qh, khj)) / math.sqrt(head_dim) for khj in kh] # compute the attention scores for the current head
            attention_weights = softmax(attention_scores) # compute the attention weights for the current head
            head_output = [sum(aw * vhi for aw, vhi in zip(attention_weights, vhj)) for vhj in zip(*vh)] # compute the output of the current head as a weighted sum of the value vectors
            x_attention.extend(head_output) # concatenate the outputs of all heads to get the output of the MHA block
        x_attention = linear(x_attention, state_dict[f'blocks.{i}.attn.wo']) # apply the output projection to the concatenated head outputs
        x = [a + b for a, b in zip(x_attention, x_residual)] # add the residual connection to get the output of the MHA block
        # mlp block
        x_residual = x # save the input to the MLP block for the residual connection
        x = rmsnorm(x) # apply RMSNorm to the input of the MLP block
        x = linear(x, state_dict[f'blocks.{i}.mlp_fc1']) # apply the first linear layer of the MLP block
        x = [xi.relu() for xi in x] # apply the ReLU activation function to the output of the first linear layer
        x = linear(x, state_dict[f'blocks.{i}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)] # add the residual connection to get the output of the MLP block
    logits = linear(x, state_dict['lm_head']) # apply the final linear layer to get the logits for the next token prediction
    return logits


# Adam optimiser
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # initialize the first moment vector for Adam
v = [0.0] * len(params) # initialize the second moment vector for Adam

# Training loop
num_steps = 20
for step in range(num_steps):
    # forward pass
    total_loss = Value(0.0)

    batch_size = 8
    for b in range(batch_size):
        # take a document from the dataset and convert it to a list of token IDs (add BOS token at the beginning)
        doc = docs[(step * batch_size + b) % len(docs)]
        tokens = [BOS] + [chars.index(c) for c in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]

        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            losses.append(-probs[target_id].log())
        total_loss += sum(losses) / n

    loss = total_loss / batch_size
    loss.backward() # compute the gradients of the loss with respect to all parameters in the computational (backward pass)

    # update the parameters using Adam (optimiser.step())
    learning_rate_target = learning_rate * (1 - step / num_steps) # linearly decay the learning rate to zero over the course of training
    for i, param in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * param.grad # update the first moment vector for the current parameter
        v[i] = beta2 * v[i] + (1 - beta2) * param.grad**2 # update the second moment vector for the current parameter
        m_hat = m[i] / (1 - beta1**(step + 1)) # compute the bias-corrected first moment estimate
        v_hat = v[i] / (1 - beta2**(step + 1)) # compute the bias-corrected second moment estimate
        param.data -= learning_rate_target * m_hat / (math.sqrt(v_hat) + eps_adam) # update the parameter using the Adam update rule
        param.grad = 0 # reset the gradient of the parameter to zero for the next iteration

    print(f"Step {step + 1}/{num_steps}, Loss: {loss.data:.4f}")

# inference (no_grad())
temperature = 0.5 # control creativity of generated text
print("\nGenerated names:")
for _ in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)] # reset the list of key and value vectors for each transformer block
    token_id = BOS # start with the BOS token
    generated_name = ''
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs] # re-normalize the probabilities
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: # stop generating if we sample the BOS token again
            break
        generated_name += chars[token_id] # append the generated character to the name
    print(generated_name)
```

<img src="/assets/images/interview/microgpt_cheatsheet.png" alt="MicroGPT quick reference cheat sheet - architecture flow, parameter count breakdown, key functions, training recipe, attention math, Adam update" style="max-width:100%">
*The complete MicroGPT architecture on one page. Every concept maps 1:1 to a line in the code above.*

> I also sometimes ask silly questions like speed of light, what is 13 in binary etc.. just to check their fluency with these numbers and in general fluency with computers is what gets you hired, not passion.

---