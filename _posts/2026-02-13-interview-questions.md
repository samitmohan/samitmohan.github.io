---
layout: post
title: "favourite interview questions"
date: 2026-02-13 14:06:04 +0530
categories: tech
tokens: "~6.5k"
---

These are some of my favourite questions to ask in interviews and I wish we move forward and have these questions in interview rounds instead of cramming algorithmic questions and deriving kadane's algorithm in 30 minutes, pretending we have never seen it before.

Fluency in computers is what gets you hired. Do you get excited when you see these questions? That's passion.

1. Computers in general - binary, speed of light, complexity
2. DSA - matrix multiplication and how to optimise it
3. Math - estimating Pi with Monte Carlo
4. Networking - what happens when you type google.com
5. Modern ML - GPT from scratch in 200 lines
6. GPU + OS - CUDA and Flash Attention

## Question 1 (covers computers in general)

> What is 13 in binary? What is the speed of light? What is merge sort complexity?

If you don't say 1101; 3 * 10^8; O(nlogn) in under 10 seconds; what the hell were you doing in college?

## Question 2 (covers dsa)

> Whats the complexity of matrix multiplication; Write matrix multiplication; How can you optimise it?

First you check if they can even be multiplied. Then you create an output array to store the results. Then you run a triple loop

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

# now try the naive python triple loop with n=500 and go make yourself a coffee
```

## Question 3 (covers math)

> Given a function that generates a random number between 0 and 1 (uniformly distributed), calculate Pi.

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

The beauty of this question: it tests geometry, probability, and coding in one shot. The candidate who derives this from first principles - not recites it from memory - actually understands what area means.

## Question 4 (covers fundamentals / networking + computer org)

> What happens when you enter www.google.com and press enter?

This is THE networking question. Every layer of the stack lights up. If you can walk through this end to end without hesitating, you understand how the internet actually works.

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

## Question 5 (covers modern ml)

> Implement a GPT from scratch. No PyTorch. No TensorFlow. Just Python and math.

This is the big one. Can you write a Transformer in 200 lines of pure Python? No `nn.Module`, no `loss.backward()` magic, no CUDA kernels. Just the raw math: scalar autograd, attention, softmax, Adam. If you can do this, you don't just know how to *use* a Transformer - you know how to *build* one.

## Question 6 (covers GPU + os)

> How does CUDA work? What is Flash Attention?

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

---

Questions 5 and 6 deserve more than a one-liner. The rest of this post is the answer: building a GPT from scratch and understanding what makes it fast on a GPU. Explaining the algorithm behind Transformers - the reason why your job will be gone - in 200 lines.

> **Note:** Andrej Karpathy recently published a [brilliant breakdown](https://karpathy.github.io/2026/02/12/microgpt/) of MicroGPT. What follows is a technical companion - a deep dive into the scalar math that makes it all work.

## The Interviewer’s Perspective

MicroGPT is my favorite interview question. When I ask a candidate to "implement a Transformer from scratch," I'm not looking for `import torch`. I'm looking for:

1. **The DAG intuition**: Do they understand that `backward()` is just a reverse topological sort?
2. **Causality at the scalar level**: Can they explain why we process tokens sequentially in this implementation instead of using a 2D mask?
3. **The Softmax stability trick**: Do they know why we subtract `max(logits)`? (Hint: it's not just "best practice," it's the difference between `inf` and a result).

## The Autograd Engine: `Value` vs. `torch.Tensor`

At the heart of MicroGPT is the `Value` class, a scalar-based automatic differentiation engine.

![Backpropagation Mechanics](/assets/images/math/backprop.png)
*The Chain Rule in action. Every node in MicroGPT tracks its children and its contribution to the final loss.*

| Feature | MicroGPT (`Value`) | PyTorch (`torch.Tensor`) |
| :--- | :--- | :--- |
| **Granularity** | **Scalar-based**. Every single number is an object. | **Tensor-based**. Operates on n-dimensional arrays. |
| **Backprop** | Manual topological sort and chain rule. | Highly optimized C++/CUDA kernels for DAG traversal. |
| **Visibility** | You can see every gradient flow through every node. | Hidden behind the `loss.backward()` black box. |

In PyTorch, we write `x = torch.randn(10, 10, requires_grad=True)`. In MicroGPT, we  initialize a list of 100 individual `Value` objects. This makes the **Chain Rule** visceral: calling `backward()` literally traverses the history of every addition and multiplication.

### The Basics: What is a Gradient?

If you've forgotten your multivariable calculus: a gradient is just a "nudge." If a weight has a gradient of `0.5`, it means that if we increase that weight by a tiny amount, the loss will increase by half that amount. Our goal is to nudge every weight in the *opposite* direction of its gradient to minimize the loss.

## Wiring vs. Modules

MicroGPT doesn't use `nn.Module`. Instead, it uses raw Python list comprehensions to implement the math.

![Embeddings](/assets/images/math/embeddings.png)
*Turning discrete tokens into high-dimensional vectors.*

- **Linear Layers**: Instead of `nn.Linear(16, 16)`, MicroGPT uses:
  `[sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]`
- **Activation**: Instead of `nn.ReLU()`, it's a manual `max(0, x)` wrapper inside the `Value` class.
- **Normalisation**: `rmsnorm(x)` is a custom function calculating the root mean square of a list of scalars manually.

## Sequential Training and the KV Cache

![Attention Mechanism](/assets/images/math/how_attention_works.png)
*How queries, keys, and values interact to create context.*

This is the most significant departure from standard PyTorch training.

- **PyTorch Way**: We train on sequences in **parallel**. We pass a batch of `[Batch, Seq_Len]` and use a **Causal Mask** (a triangular matrix of $-\infty$) to hide future tokens.
- **MicroGPT Way**: It processes tokens **one by one** (`for pos_id in range(n)`). It appends keys and values to a list (a **KV Cache**) as it goes.

**Why?** Implementing a 2D causal mask and matrix multiplication using scalar `Value` objects would be catastrophically slow. By using a KV cache during training, Karpathy makes "causality" (not looking ahead) implicit and the code much easier to read.

## Adam from Scratch

![Adam Optimizer](/assets/images/math/adam.png)
*Adam uses momentum and variance to navigate the loss landscape.*

In PyTorch, we call `optimizer.step()`. MicroGPT manually tracks:

1. `m`: The first moment (momentum).
2. `v`: The second moment (uncentered variance).
3. `m_hat` / `v_hat`: Bias correction for the early steps of training.

It then updates `param.data` directly. This proves that Adam is just an adaptive learning rate that scales every single weight update based on its own history.

## The Gradient Journey

![Gradients](/assets/images/math/gradients.png)

When we call `loss.backward()`, we are executing a "Chain Reaction" of local derivatives. 

Imagine the very last operation: `loss = total_loss / batch_size`.

- The `loss.grad` is seeded at `1.0`.
- This flows back to `total_loss.grad` as `1.0 / batch_size`.
- If `total_loss = loss1 + loss2`, then both `loss1.grad` and `loss2.grad` inherit `total_loss.grad` (because the derivative of `x+y` w.r.t `x` is `1`).

This continues until we reach the **Token Embeddings**. In MicroGPT, we can literally print `state_dict['wte'][10][5].grad` to see how much the 5th dimension of the 10th token's embedding contributed to the error of a specific name like "Andre-j".

## Numerical Stability: The `max_logit` Trick

![Safe Softmax](/assets/images/ai/safesoftmax.png)

In the `softmax` function, you'll see:
`max_logit = max(l.data for l in logits)`
`exp_logits = [(logit - max_logit).exp() for logit in logits]`

**Why?**
The exponential function `e^x` grows catastrophically fast. If a logit is `100`, `e^100` is ~2.6e43, which will overflow a floating-point number. By subtracting the maximum value, the largest value becomes `e^0 = 1`, and everything else becomes a small fraction between `0` and `1`. Since softmax is translation-invariant ($Softmax(x) = Softmax(x - c)$), the math remains identical, but the code stops crashing.

## Scaling to Production

You might wonder: "If we can write a GPT in 200 lines of Python, why is PyTorch so huge?"

The answer is **Vectorization** and **Kernel Fusion**. 
In MicroGPT, calculating a dot product involves a Python `for` loop, which is slow. In production, we use NVIDIA's CUDA or OpenAI's Triton to launch thousands of threads that compute these products in parallel. 

When you see `torch.matmul(A, B)`, the engine isn't just looping; it's using "Tiling" to move chunks of data into the GPU's fast SRAM (Static Random-Access Memory), computing the result, and moving it back. MicroGPT is the *logic*; PyTorch is the *plumbing*.

## Conclusion

While Karpathy's own breakdown is the definitive source, I hope this deep dive into the scalar mechanics, the interview perspective, and the "Tensor Bridge" has given you a more visceral understanding of how LLMs actually work. 

The code below is my own implementation—clocking in at just over 200 lines. It is designed to be read line-by-line, without the abstraction of modern deep learning frameworks. If you can walk through this code and explain every gradient, you don't just know how to *use* a Transformer; you know how to *build* one.

Check out Andrej's original blog [here](https://karpathy.github.io/2026/02/12/microgpt/) for the high-level context.

---

## The Code (200 Lines)

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
head_dim = n_embed // n_head # dimnension of each attention head

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
