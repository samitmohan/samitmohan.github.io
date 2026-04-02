---
layout: post
title: "building pytorch from scratch"
date: 2026-03-11 14:06:04 +0530
categories: tech
tokens: "~18k"
math: true
description: "Building pytorch from scratch in ~1300 lines - reverse-mode autograd, conv2d, optimizers. Trains MNIST. No C++, no CUDA, just closures and numpy."
---

> **TL;DR:** PyTorch operations build a computation graph by attaching _backward closures to tensors. .backward() topologically sorts the graph and fires each closure in reverse. 

This started when I wanted to learn deep learning, first step was to learn PyTorch; only way to learn something is to build it from scratch.

*Prerequisites:* python, calculus (chain rule), some pytorch. if not, read [neural networks](/tech/2025/10/25/nn.html), [tensor library](/tech/2026/01/07/torch.html), and [backprop math](/tech/2026/01/21/math.html) first.

Docs: [minitorch documentation, examples and source code](https://samitmohan.github.io/minitorch/)

---

<style>
*{box-sizing:border-box;margin:0;padding:0}
.wrap{font-family:var(--font-sans);color:var(--color-text-primary);max-width:680px}
.hero{margin-bottom:1.5rem}
.hero h2{font-size:22px;font-weight:500;margin-bottom:4px}
.hero p{font-size:13px;color:var(--color-text-secondary);line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:1.5rem}
.stat{background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:12px 14px}
.stat .label{font-size:11px;color:var(--color-text-tertiary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.stat .val{font-size:22px;font-weight:500}
.chart-section{margin-bottom:1.5rem}
.chart-section h3{font-size:16px;font-weight:500;margin-bottom:10px}
.chart-wrap{position:relative;width:100%;height:260px}
.modules{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1.5rem}
.mod-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:14px 16px;cursor:pointer;transition:border-color .15s}
.mod-card:hover{border-color:var(--color-border-secondary)}
.mod-card.active{border-color:var(--color-border-info);border-width:2px}
.mod-name{font-size:14px;font-weight:500;margin-bottom:2px}
.mod-lines{font-size:12px;color:var(--color-text-secondary)}
.mod-desc{font-size:12px;color:var(--color-text-secondary);margin-top:6px;line-height:1.5;display:none}
.mod-card.active .mod-desc{display:block}
.mod-concepts{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;display:none}
.mod-card.active .mod-concepts{display:flex}
.pill{font-size:11px;padding:3px 8px;border-radius:var(--border-radius-md);background:var(--color-background-info);color:var(--color-text-info)}
.pill.g{background:var(--color-background-success);color:var(--color-text-success)}
.pill.w{background:var(--color-background-warning);color:var(--color-text-warning)}
.pill.d{background:var(--color-background-danger);color:var(--color-text-danger)}
.section-title{font-size:16px;font-weight:500;margin-bottom:10px}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:1.5rem}
.flow-step{font-size:12px;padding:6px 12px;border-radius:var(--border-radius-md);background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);cursor:pointer;transition:all .15s}
.flow-step:hover{border-color:var(--color-border-secondary)}
.flow-step.hl{background:var(--color-background-info);color:var(--color-text-info);border-color:transparent}
.flow-arrow{font-size:14px;color:var(--color-text-tertiary)}
.detail-box{background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:14px 16px;font-size:13px;color:var(--color-text-secondary);line-height:1.6;min-height:60px;margin-bottom:1.5rem}
.results{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.res-card{background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);padding:14px 16px}
.res-card .model{font-size:14px;font-weight:500;margin-bottom:2px}
.res-card .acc{font-size:22px;font-weight:500;color:var(--color-text-success);margin-bottom:2px}
.res-card .meta{font-size:12px;color:var(--color-text-secondary);line-height:1.5}
</style>
<div class="wrap">
  <div class="hero">
    <h2>minitorch - building pytorch from scratch</h2>
    <p>Reverse-mode autograd, module system, conv2d with im2col, optimizers. ~1,300 lines of Python + NumPy. Trains MNIST to 96%.</p>
  </div>

  <div class="stats">
    <div class="stat"><div class="label">Total lines</div><div class="val">~1,300</div></div>
    <div class="stat"><div class="label">Files</div><div class="val">11</div></div>
    <div class="stat"><div class="label">Dependencies</div><div class="val">NumPy</div></div>
    <div class="stat"><div class="label">Best accuracy</div><div class="val">96.4%</div></div>
  </div>

  <div class="chart-section">
    <h3>Codebase breakdown</h3>
    <div class="chart-wrap"><canvas id="codeChart"></canvas></div>
  </div>

  <div class="section-title">Module explorer</div>
  <p style="font-size:12px;color:var(--color-text-tertiary);margin-bottom:10px">Click a module to see details and key concepts</p>
  <div class="modules" id="modules"></div>

  <div class="section-title">Training loop flow</div>
  <div class="flow" id="flow"></div>
  <div class="detail-box" id="flow-detail">Click a step above to see what happens at each stage of the training loop.</div>

  <div class="section-title" style="margin-top:0.5rem">MNIST results</div>
  <div class="results">
    <div class="res-card">
      <div class="model">MLP (2-layer)</div>
      <div class="acc">95.15%</div>
      <div class="meta">784 → 128 → 10<br>15 epochs, 10k samples<br>Adam optimizer</div>
    </div>
    <div class="res-card">
      <div class="model">CNN (2 conv blocks)</div>
      <div class="acc">96.40%</div>
      <div class="meta">Conv→Pool→Conv→Pool→FC<br>10 epochs, 2k samples<br>Adam optimizer</div>
    </div>
  </div>
</div>

<script>
lazyWidget('codeChart', function() {
  var s = document.createElement('script');
  s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
  s.onload = function() {
const modules = [
  {name:'tensor.py',lines:522,desc:'The core: Tensor class with all operators and autograd engine. Every op builds a computation graph via _backward closures. .backward() topologically sorts and fires closures in reverse.',concepts:['Computation graph','Topological sort','_backward closures','Broadcasting gradients','_accum_grad'],cat:''},
  {name:'module.py',lines:140,desc:'Module system and Sequential container. Recursively collects parameters via __dict__ introspection. state_dict/load_state_dict for serialization. train/eval mode toggling.',concepts:['parameters()','Sequential','state_dict','train/eval mode'],cat:'g'},
  {name:'layers.py',lines:109,desc:'Linear layer with Kaiming/Xavier init, ReLU, Dropout (inverted scaling), and BatchNorm1d with running statistics. BatchNorm uses autograd ops - no hand-written backward needed.',concepts:['Kaiming init','Inverted dropout','BatchNorm1d','Running statistics'],cat:''},
  {name:'conv.py',lines:150,desc:'Conv2d via im2col trick - converts convolution into matrix multiply. col2im for backward. MaxPool2d uses NumPy stride_tricks for zero-copy patch extraction.',concepts:['im2col / col2im','Stride tricks','MaxPool2d mask','Flatten'],cat:'w'},
  {name:'functional.py',lines:82,desc:'Activation functions: relu, sigmoid, tanh, softmax, log_softmax. All follow the same pattern - forward with NumPy, backward closure with chain rule. Numerical stability via max-subtraction.',concepts:['ReLU mask gating','Log-sum-exp trick','Numerical stability'],cat:'g'},
  {name:'loss.py',lines:25,desc:'MSE (one-liner), BCE with clamping, cross-entropy with one-hot conversion and fused log_softmax for stability.',concepts:['MSE','BCE','Cross-entropy','One-hot encoding'],cat:''},
  {name:'optim.py',lines:113,desc:'SGD with momentum (velocity accumulation), Adam with bias-corrected first/second moments. Gradient clipping (norm and value). StepLR and cosine annealing schedulers.',concepts:['Momentum','Adam moments','Bias correction','Gradient clipping','LR scheduling'],cat:'w'},
  {name:'grad_check.py',lines:56,desc:'Numerical gradient verification via central finite differences in float64. Compares analytic vs numerical gradients with np.allclose. Catches silently wrong backward implementations.',concepts:['Central differences','Float64 precision','np.allclose'],cat:'d'},
  {name:'viz.py',lines:39,desc:'Renders computation graph with graphviz. Green nodes = requires_grad, gray = constants, blue boxes = operations. Useful for debugging missing edges or detached tensors.',concepts:['DAG visualization','Graphviz'],cat:''},
  {name:'data.py',lines:27,desc:'Minimal DataLoader - shuffles indices and yields batches of tensors. No multiprocessing, no prefetch. Just enough to feed a training loop.',concepts:['Batch iteration','Index shuffling'],cat:'g'},
  {name:'backend.py',lines:33,desc:'Optional CuPy GPU backend with get_array_module() dispatch. Transparent switch between NumPy and CuPy arrays. im2col still CPU-bound.',concepts:['CuPy dispatch','Array module pattern'],cat:''},
];

const flowSteps = [
  {label:'Forward pass',detail:'Input flows through the model - each layer computes its output and records a _backward closure capturing the chain rule derivative. This builds the computation graph as a DAG of tensors connected by operations.'},
  {label:'Compute loss',detail:'The loss function (MSE, cross-entropy) produces a scalar tensor. This scalar is the tip of the computation graph - calling .backward() on it will propagate gradients through every operation that created it.'},
  {label:'zero_grad()',detail:'Clear all parameter gradients from the previous step. Without this, gradients accumulate across iterations (useful for gradient accumulation, but wrong by default).'},
  {label:'.backward()',detail:'Seeds loss.grad = 1.0, topologically sorts the graph (depth-first), then walks it in reverse calling each node\'s _backward() closure. Gradients flow from loss back to every leaf parameter via the chain rule.'},
  {label:'optimizer.step()',detail:'Uses the accumulated gradients to update parameters. SGD: theta -= lr * grad. Adam: tracks running mean and variance of gradients, applies bias correction, then updates with adaptive per-parameter learning rates.'},
];

const container = document.getElementById('modules');
modules.forEach((m,i) => {
  const card = document.createElement('div');
  card.className = 'mod-card';
  card.innerHTML = `<div class="mod-name">${m.name}</div><div class="mod-lines">${m.lines} lines</div><div class="mod-desc">${m.desc}</div><div class="mod-concepts">${m.concepts.map(c=>`<span class="pill ${m.cat}">${c}</span>`).join('')}</div>`;
  card.onclick = () => {
    document.querySelectorAll('.mod-card').forEach(c=>c.classList.remove('active'));
    card.classList.toggle('active');
  };
  container.appendChild(card);
});

const flowEl = document.getElementById('flow');
flowSteps.forEach((s,i) => {
  if(i>0){const a=document.createElement('span');a.className='flow-arrow';a.textContent='→';flowEl.appendChild(a)}
  const step = document.createElement('span');
  step.className = 'flow-step';
  step.textContent = s.label;
  step.onclick = () => {
    document.querySelectorAll('.flow-step').forEach(e=>e.classList.remove('hl'));
    step.classList.add('hl');
    document.getElementById('flow-detail').textContent = s.detail;
  };
  flowEl.appendChild(step);
});

const dark = matchMedia('(prefers-color-scheme:dark)').matches;
const colors = ['#534AB7','#0F6E56','#D85A30','#993556','#185FA5','#639922','#BA7517','#5F5E5A','#E24B4A','#1D9E75','#3C3489'];
const borderColors = dark ? colors.map(c=>c+'cc') : colors;

new Chart(document.getElementById('codeChart'), {
  type:'bar',
  data:{
    labels: modules.map(m=>m.name.replace('.py','')),
    datasets:[{
      data: modules.map(m=>m.lines),
      backgroundColor: colors.map(c=> c + (dark?'aa':'cc')),
      borderColor: borderColors,
      borderWidth:1,
      borderRadius:4
    }]
  },
  options:{
    indexAxis:'y',
    responsive:true,
    maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>ctx.raw+' lines'}}},
    scales:{
      x:{grid:{color:dark?'rgba(255,255,255,0.06)':'rgba(0,0,0,0.06)'},ticks:{color:dark?'#9c9a92':'#73726c',font:{size:11}}},
      y:{grid:{display:false},ticks:{color:dark?'#c2c0b6':'#3d3d3a',font:{size:11,family:'"Anthropic Sans",sans-serif'}}}
    }
  }
});
  };
  document.head.appendChild(s);
});
</script>

---

## 1. what's in the box

- `tensor.py` (522 lines) - the core: Tensor class, all operators, autograd
- `module.py` (140 lines) - the Module system, Sequential container
- `layers.py` (109 lines) - Linear, ReLU, Dropout, BatchNorm
- `conv.py` (150 lines) - Conv2d, MaxPool2d, im2col/col2im
- `functional.py` (82 lines) - activation functions
- `loss.py` (25 lines) - MSE, BCE, cross-entropy
- `optim.py` (113 lines) - SGD, Adam, gradient clipping, LR schedulers
- `grad_check.py` (56 lines) - numerical gradient verification
- `viz.py` (39 lines) - computation graph visualization
- `data.py` (27 lines) - DataLoader
- `backend.py` (33 lines) - optional CuPy GPU support

~1,300 lines total. This can be fairly reduced but I am too lazy. Let's walk through it.

---

## 2. the tensor

A tensor is a multi-dimensional array that optionally tracks operations so gradients can be computed later. 

```python
class Tensor:
    def __init__(self, data, *, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        elif isinstance(data, np.floating):
            self.data = np.array(data, dtype=data.dtype)
        elif gpu_available() and hasattr(data, '__cuda_array_interface__'):
            self.data = data if data.dtype in (np.float32, np.float64) else data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad and _grad_enabled
        self.grad = None
        self._backward = lambda: None
        self._prev = set()
        self._op = ''
```

Five fields do all the work:

- **`data`**: a NumPy array holding the actual values. Make everything to float32 (or float64) because you can't differentiate integers.
- **`requires_grad`**: a boolean flag. If `True`, track operations on this tensor. Notice it's AND'd with `_grad_enabled` - more on that in a moment.
- **`grad`**: where the gradient ends up after `.backward()`. Starts as `None`, gets lazily initialized.
- **`_backward`**: a closure. Every operation that creates a new tensor stuffs a function in here that knows how to push gradients back to the inputs. We initialize it to a no-op lambda.
- **`_prev`**: the set of parent tensors that were used to create this one. Together with `_backward`, this forms the **computation graph**.
- **`_op`**: a string label for debugging and visualization ('add', 'mul', 'matmul', etc.).

Static constructors wrap NumPy:

```python
@staticmethod
def zeros(*shape, requires_grad=False, device="cpu"):
    t = Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)
    return t.to(device) if device != "cpu" else t

@staticmethod
def randn(*shape, requires_grad=False, device="cpu"):
    t = Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)
    return t.to(device) if device != "cpu" else t
```

Then there's the **`no_grad`** context manager. During inference, you don't want to build a computation graph; it wastes memory and you'll never call backward. So we have a global flag:

```python
_grad_enabled = True

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev
```

When you write `with no_grad():`, it flips the global to `False`. Now any new tensor created inside the block has `requires_grad=False` regardless of what you asked for, because of the `requires_grad and _grad_enabled` check in `__init__`. On exit, it restores the previous state. Smart no? I could never come up with this elegance myself; this is exactly what PyTorch does.

---

## 3. the autograd engine

The core of the whole thing. Everything else is filling in operators.

### the computation graph

Every arithmetic operation on tensors doesn't just compute a result - it records *how* to undo itself. When you write `c = a + b`, tensor `c` carries two things:

1. The numerical result (`c.data = a.data + b.data`)
2. A closure (`c._backward`) that knows: "if someone gives me the gradient of the loss with respect to `c`, here's how to compute the gradients with respect to `a` and `b`"

The tensors and their `_prev` links form a **directed acyclic graph** (DAG). The leaf tensors (your weights and inputs) are at the roots. The loss scalar is at the tip. Every intermediate result is a node in between, connected by the operations that created it.

### walking through an example

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/minitorch/autograd_backward_poster.jpg">
  <source src="/assets/images/minitorch/autograd_backward.mp4" type="video/mp4">
</video>

Let's trace through this step by step:

```python
a = Tensor([1.0, 2.0], requires_grad=True)
b = a ** 2        # b = [1, 4]
c = b.sum()       # c = 5.0
c.backward()
# a.grad = [2.0, 4.0]
```

**Step 1: `a = Tensor([1.0, 2.0], requires_grad=True)`**

This creates a leaf tensor. No `_backward`, no `_prev`. It's a starting point.

**Step 2: `b = a ** 2`**

This calls `__pow__`. Looking at the actual code:

```python
def __pow__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data ** other.data, requires_grad=self._should_track(other))

    def _backward():
        if self.requires_grad:
            # d(a^b)/da = b * a^(b-1)
            _accum_grad(self, _sum_to_shape(other.data * (self.data ** (other.data - 1)) * out.grad, self.data.shape))
        if other.requires_grad:
            # d(a^b)/db = a^b * ln(|a|)
            safe_base = np.abs(self.data) + 1e-12
            _accum_grad(other, _sum_to_shape(out.data * np.log(safe_base) * out.grad, other.data.shape))

    out._backward = _backward
    out._prev = {self, other}
    out._op = 'pow'
    return out
```

The forward pass computes `[1.0, 2.0] ** 2 = [1.0, 4.0]`. But it also creates a closure that captures `self` (which is `a`), `other` (which is the constant `2`), and `out` (which is `b`). This closure knows the derivative of $x^n$ is $nx^{n-1}$.

So `b._backward` is a function that, when called, will compute:

$$\frac{\partial}{\partial a} a^2 = 2a \cdot \text{grad}_b = 2 \cdot [1, 2] \cdot \text{grad}_b$$

And `b._prev = {a, Tensor(2)}` - the parent nodes.

**Step 3: `c = b.sum()`**

```python
def sum(self, axis=None, keepdims=False):
    data = self.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(data, requires_grad=self.requires_grad and _grad_enabled)

    def _backward():
        if self.requires_grad:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            _accum_grad(self, np.broadcast_to(grad, self.data.shape))

    out._backward = _backward
    out._prev = {self}
    out._op = 'sum'
    return out
```

Sum reduces `[1, 4]` to `5.0`. The backward closure broadcasts the scalar gradient back to the original shape. The derivative of `sum` with respect to each input element is 1, so the gradient just gets copied to every position.

**Step 4: `c.backward()`**

Here's `backward()`:

```python
def backward(self):
    assert self.data.size == 1, "backward() only works on scalar tensors"
    if self.grad is None:
        self.grad = np.ones_like(self.data)

    topo = []
    visited = set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(self)

    for node in reversed(topo):
        node._backward()
```

Three things happen:

1. **Seed the gradient**: `c.grad = 1.0`. The derivative of a scalar with respect to itself is 1.
2. **Topological sort**: Walk the graph depth-first, appending each node after all its children. This gives us the order `[a, Tensor(2), b, c]`.
3. **Reverse pass**: Walk the sorted list backwards: `c, b, a, Tensor(2)`. For each node, call its `_backward()`.

When `c._backward()` fires, it sets `b.grad = broadcast(1.0, [2]) = [1.0, 1.0]`.

When `b._backward()` fires, it computes `a.grad = 2 * [1, 2] * [1, 1] = [2.0, 4.0]`.

And we're done. `a.grad = [2.0, 4.0]`, which is correct: $\frac{d}{da}(a_1^2 + a_2^2) = [2a_1, 2a_2] = [2, 4]$.

### the helper functions

Two small helpers make the backward closures work correctly.

**`_accum_grad`** handles the fact that a tensor might be used multiple times in a computation (like `a * a`). Each usage contributes to the gradient, so we *accumulate* rather than overwrite:

```python
def _accum_grad(tensor, grad):
    if tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data)
    tensor.grad += grad
```

**`_sum_to_shape`** handles **broadcasting gradients**. When NumPy broadcasts a `(1, 3)` tensor against a `(4, 3)` tensor, the result is `(4, 3)`. But in the backward pass, the gradient is `(4, 3)` and we need `(1, 3)` for the smaller input. Fix: sum along the broadcasted dimensions:

```python
def _sum_to_shape(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)
```

Why does summing undo broadcasting? Broadcasting copies a value to multiple positions in the forward pass. That single value *contributed* to multiple outputs. In the backward pass, we sum those contributions back into one gradient. Multi-variable chain rule.

My first version didn't handle broadcasting at all - gradients just passed through with whatever shape they had. Training "worked" in the sense that loss went down, but the gradients were wrong. No error, just silently wrong numbers. The model would converge to mediocre accuracy and I couldn't figure out why. It wasn't until I ran gradient checking (section 11) that I saw the shapes were mismatched. The fix is three lines of code but an entire evening for me. Should've used claude.

### three operators in detail

Every operator follows the same pattern. Discussing the three common ones:

**Addition** is the simplest. The derivative of $a + b$ with respect to both $a$ and $b$ is 1, so the gradient just passes through (after handling broadcasting):

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, requires_grad=self._should_track(other))

    def _backward():
        if self.requires_grad:
            _accum_grad(self, _sum_to_shape(out.grad, self.data.shape))
        if other.requires_grad:
            _accum_grad(other, _sum_to_shape(out.grad, other.data.shape))

    out._backward = _backward
    out._prev = {self, other}
    out._op = 'add'
    return out
```

**Multiplication** is straightforward. The derivative of $a \cdot b$ with respect to $a$ is $b$, and with respect to $b$ is $a$. Each input's gradient uses the *other* input's value:

```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, requires_grad=self._should_track(other))

    def _backward():
        if self.requires_grad:
            _accum_grad(self, _sum_to_shape(other.data * out.grad, self.data.shape))
        if other.requires_grad:
            _accum_grad(other, _sum_to_shape(self.data * out.grad, other.data.shape))

    out._backward = _backward
    out._prev = {self, other}
    out._op = 'mul'
    return out
```

**Matrix multiply** is the tricky one. If $C = A @ B$ where $A$ is $(m, k)$ and $B$ is $(k, n)$, then:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T \qquad \frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$

This falls out of the chain rule applied to matrix calculus. The shapes work out perfectly:

- `out.grad` is $(m, n)$
- `other.data.T` is $(n, k)$
- So `out.grad @ other.data.T` is $(m, k)$ - same shape as `self`

```python
def __matmul__(self, other):
    out = Tensor(self.data @ other.data, requires_grad=self._should_track(other))

    def _backward():
        if self.requires_grad:
            _accum_grad(self, out.grad @ other.data.T)
        if other.requires_grad:
            _accum_grad(other, self.data.T @ out.grad)

    out._backward = _backward
    out._prev = {self, other}
    out._op = 'matmul'
    return out
```

The gradient of `A @ B` with respect to `A` is `grad @ B.T`, and with respect to `B` is `A.T @ grad`. This exact pattern shows up again in the Conv2d backward pass.

---

## 4. operations

Every operation in minitorch follows the same template:

1. Compute the forward result using NumPy
2. Define a `_backward` closure that uses the chain rule
3. Set `out._backward`, `out._prev`, `out._op`
4. Return the new tensor

Once you've seen this pattern three times, you've seen it forty times. Here's ReLU from `functional.py`:

```python
def relu(x):
    data = np.maximum(0, x.data)
    out = Tensor(data, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            _accum_grad(x, (x.data > 0) * out.grad)

    out._backward = _backward
    out._prev = {x}
    out._op = 'relu'
    return out
```

`(x.data > 0)` is a boolean mask that gates the gradient. Same structure as everything else: compute forward, capture what you need in the closure, define backward using the chain rule.

Sigmoid, tanh, softmax, log-softmax - all follow the same template. The only interesting wrinkle is **numerical stability**: naive softmax overflows to `inf` for any input greater than ~88 (float32 limit for `exp()`). My first forward pass produced NaN loss. The fix is subtracting the max before exponentiating - shift-invariant, keeps everything in safe range. Same idea shows up in log-softmax where you fuse the log and softmax to avoid the precision-destroying round trip of `log(exp(x))`.

Here's the full catalog:

| Category | Operations |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `**`, unary `-`, `@` (matmul) |
| Reductions | `sum`, `mean`, `var`, `std`, `max`, `min` |
| Shape | `reshape`, `transpose`, `squeeze`, `unsqueeze`, `__getitem__` |
| Elementwise | `exp`, `log`, `clamp`, `abs` |
| Activations | `relu`, `sigmoid`, `tanh`, `softmax`, `log_softmax` |
| Tensor ops | `cat`, `stack` |

Every one has a hand-written backward closure. You can chain any combination and the backward pass works, because the chain rule composes.

---

## 5. the module system

Tensors with autograd need organizing into layers and models. That's the **Module** system.

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/minitorch/module_system_poster.jpg">
  <source src="/assets/images/minitorch/module_system.mp4" type="video/mp4">
</video>
*Module hierarchy, recursive parameter collection, and state_dict serialization.*

```python
class Module:
    def __init__(self):
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

`__call__` -> `forward` indirection so subclasses override `forward` while `__call__` handles bookkeeping (hooks, profiling in real PyTorch).

**`parameters()`** recursively collects all trainable tensors by introspecting `__dict__`:

```python
def parameters(self):
    params = []
    for val in self.__dict__.values():
        if isinstance(val, Tensor) and val.requires_grad:
            params.append(val)
        elif isinstance(val, Module):
            params.extend(val.parameters())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Tensor) and item.requires_grad:
                    params.append(item)
                elif isinstance(item, Module):
                    params.extend(item.parameters())
    return params
```

If you assign a `Tensor` with `requires_grad=True` to `self.whatever`, it gets found. Same for child `Module`s. It walks lists and tuples too, which is how `Sequential` stores its layers.

The rest is what you'd expect: `state_dict()` / `load_state_dict()` for serialization via `np.savez`, `train()` / `eval()` to toggle behavior for Dropout and BatchNorm, and a `Sequential` container that's just a for-loop over layers.

---

## 6. layers

### linear

Matrix multiply plus bias:

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init='kaiming'):
        super().__init__()
        if init == 'xavier':
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            (np.random.randn(in_features, out_features) * scale).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        ) if bias else None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y
```

The autograd engine handles the backward pass automatically - `@` and `+` already have backward closures.

The weight initialization matters more than you'd think. I first tried `np.random.randn` with no scaling. The MLP "trained"; 60% accuracy on MNIST. It was that activations were dying in the forward pass because the initial weights were way too large for ReLU. The fix:

- **Kaiming init**: scale by $\sqrt{2/\text{fan\_in}}$. The factor of 2 compensates for ReLU killing half the signal.
- **Xavier init**: scale by $\sqrt{2/(\text{fan\_in} + \text{fan\_out})}$. Better for sigmoid/tanh.

Switching to Kaiming init jumped accuracy from ~60% to ~95%. Three characters of code difference (`2.0` instead of `1.0` in the numerator).

Activation layers (ReLU, Sigmoid, Tanh) are one-line wrappers around the functional versions so they can slot into `Sequential`. Nothing interesting there.

### dropout

**Dropout** randomly zeros out elements during training to prevent overfitting. The implementation uses the **inverted dropout** pattern:

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self._training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                _accum_grad(x, out.grad * mask * scale)

        out._backward = _backward
        out._prev = {x}
        out._op = 'dropout'
        return out
```

The key detail is the `1/(1-p)` scaling. This is "inverted" dropout - scale *during training* so that at test time, the network works unchanged. Without it, activations are $(1-p)$ times smaller during training than inference, and your model silently breaks at eval time. Backward applies the same mask and scale.

### BatchNorm1d

**Batch normalization** normalizes each feature across the batch, then applies a learnable scale and shift:

```python
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        if self._training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff ** 2).mean(axis=0, keepdims=True)
            x_hat = diff / (var + self.eps) ** 0.5

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.flatten()
        else:
            rm = Tensor(self.running_mean.reshape(1, -1))
            rv = Tensor(self.running_var.reshape(1, -1))
            x_hat = (x - rm) / (rv + self.eps) ** 0.5

        return x_hat * self.gamma + self.beta
```

During training: normalize each feature by batch mean and variance, then apply learnable $\gamma$ (scale) and $\beta$ (shift). It also tracks running statistics via exponential moving average for use at eval time (where batch stats are meaningless with a single sample).

The nice thing here: no hand-written backward needed. The forward pass is built entirely from differentiable Tensor operations (mean, subtraction, division, pow), so autograd handles it automatically.

---

## 7. loss functions

The scalar you call `.backward()` on.

```python
def mse_loss(input, target):
    return ((input - target) ** 2).mean()
```

MSE is a one-liner. Gradients flow through `mean`, `**`, and `-` automatically. BCE is similar (clamp inputs to avoid `log(0)`, apply the binary cross-entropy formula). The interesting one is cross-entropy:

```python
def cross_entropy_loss(input, target):
    if target.data.ndim == 1 or (target.data.ndim == 2 and target.data.shape[1] == 1):
        labels = target.data.flatten().astype(np.int64)
        one_hot = np.zeros((input.data.shape[0], input.data.shape[1]), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0
        target = Tensor(one_hot, requires_grad=False)

    log_probs = F.log_softmax(input, axis=1)
    N = input.data.shape[0]
    return (-log_probs * target).sum() / N
```

**Cross-entropy** is more involved. It accepts either integer class labels or one-hot vectors. If you pass labels, it converts them to one-hot first. Then it uses `log_softmax` - not `log(softmax(x))`. Why? Because computing softmax first and then taking the log is numerically unstable: softmax can produce values very close to zero, and `log(~0) = -huge_number`. The `log_softmax` function fuses these two operations and uses the log-sum-exp trick to stay stable.

The formula is:

$$\text{CE} = -\frac{1}{N}\sum_i \sum_c y_{ic} \log \text{softmax}(x_{ic})$$

Since $y$ is one-hot, this simplifies to picking out the log-probability of the correct class.

---

## 8. optimizers

Takes gradients from `.backward()` and updates parameters.

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/minitorch/adam_poster.jpg">
  <source src="/assets/images/minitorch/adam.mp4" type="video/mp4">
</video>
*Adam tracking momentum and variance, parameter converging toward the optimum over 5 steps.*

### SGD with momentum

Vanilla SGD: $\theta \leftarrow \theta - \alpha \nabla L$. Momentum adds a velocity term:

$$v_t = \mu \cdot v_{t-1} + g_t$$
$$\theta_t = \theta_{t-1} - \alpha \cdot v_t$$

where $\mu$ is the momentum coefficient (typically 0.9), $g_t$ is the current gradient, and $\alpha$ is the learning rate.

```python
class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.copy()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            v[:] = self.momentum * v + grad
            p.data -= self.lr * v
```

Momentum is a ball rolling downhill. If the gradient keeps pointing the same direction, velocity builds. If it oscillates, opposing contributions cancel out.

### Adam

**Adam** (Adaptive Moment Estimation) tracks both the first moment (mean) and second moment (uncentered variance) of the gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

These estimates are biased toward zero at the start (since they're initialized to zero), so we apply bias correction:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Then update:

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            xp = get_array_module(p.data)
            grad = p.grad.copy()
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
```

Adam adapts the learning rate per-parameter: large-gradient params get smaller steps, noisy-gradient params get larger ones.

minitorch also has gradient clipping (both norm and value), StepLR, and cosine annealing schedulers. Standard stuff - the implementations are in `optim.py` if you want the details.

---

## 9. convolutions

> matrix multiplication is just convolution

Naive convolution is nested loops. Painfully slow. Every serious framework uses **im2col** instead: convert the convolution into a matrix multiply.

### the im2col trick

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/minitorch/im2col_poster.jpg">
  <source src="/assets/images/minitorch/im2col.mp4" type="video/mp4">
</video>

<video autoplay loop muted playsinline style="max-width:100%" preload="none" poster="/assets/images/minitorch/conv2d_im2col_poster.jpg">
  <source src="/assets/images/minitorch/conv2d_im2col.mp4" type="video/mp4">
</video>
*Patches extracted from input, flattened into rows, then matrix-multiplied with the kernel.*

A convolution with a $k \times k$ kernel takes a $k \times k$ patch of the input and dots it with the kernel. It does this at every valid position. What if we extracted *every* patch, flattened each into a row, and stacked them into a matrix? Then the convolution becomes a single matrix multiply.

For an input of shape $(N, C, H, W)$ with kernel size $(k_h, k_w)$:

1. Extract every $(C, k_h, k_w)$ patch at every valid spatial position
2. Flatten each patch into a row vector of length $C \cdot k_h \cdot k_w$
3. Stack all $N \cdot O_H \cdot O_W$ patches into a matrix
4. Multiply by the weight matrix (which has one row per output channel)

```python
def im2col(input_data, kh, kw, stride, padding):
    N, C, H, W = input_data.shape
    if padding > 0:
        input_data = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, _, H_pad, W_pad = input_data.shape
    OH = (H_pad - kh) // stride + 1
    OW = (W_pad - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, OH, OW), dtype=input_data.dtype)
    for y in range(kh):
        y_max = y + stride * OH
        for x in range(kw):
            x_max = x + stride * OW
            cols[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
```

The loops here are over the kernel dimensions ($k_h \times k_w$), not over the spatial dimensions. For a $3 \times 3$ kernel, that's just 9 iterations regardless of image size. Each iteration uses NumPy slicing with strides to extract all patches at one kernel position across all batch elements and channels simultaneously. The "expensive" part (spatial iteration) is handled by NumPy's vectorized strided slicing.

The final `reshape` flattens each patch into a row, giving us a matrix of shape $(N \cdot O_H \cdot O_W, C \cdot k_h \cdot k_w)$.

**col2im** is the inverse operation, used in the backward pass. It takes the column matrix and scatters it back into the spatial format, accumulating where patches overlap:

```python
def col2im(cols, input_shape, kh, kw, stride, padding):
    N, C, H, W = input_shape
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    OH = (H_pad - kh) // stride + 1
    OW = (W_pad - kw) // stride + 1

    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for y in range(kh):
        y_max = y + stride * OH
        for x in range(kw):
            x_max = x + stride * OW
            img[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img
```

Note the `+=` instead of `=`. When stride is 1, adjacent patches overlap, so the same input position contributes to multiple output positions. In the backward pass, we need to accumulate all those gradient contributions.

### Conv2d

With im2col, the Conv2d forward pass is clean:

```python
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(
            (np.random.randn(out_channels, in_channels * kh * kw) * scale).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
```

Notice the weight shape: `(out_channels, in_channels * kh * kw)`. We store the kernel *already flattened* for the matrix multiply. Kaiming initialization with `fan_in = in_channels * kh * kw`.

```python
    def forward(self, x):
        N, C, H, W = x.data.shape
        kh, kw = self.kernel_size
        OH = (H + 2 * self.padding - kh) // self.stride + 1
        OW = (W + 2 * self.padding - kw) // self.stride + 1

        cols = im2col(x.data, kh, kw, self.stride, self.padding)
        out_data = cols @ self.weight.data.T + self.bias.data
        out_data = out_data.reshape(N, OH, OW, self.out_channels).transpose(0, 3, 1, 2)
        out = Tensor(out_data, requires_grad=x.requires_grad or self.weight.requires_grad)
```

The forward pass is three lines of real work:
1. `im2col` extracts patches into a matrix
2. Matrix multiply with transposed weights, add bias
3. Reshape and transpose back to $(N, C_{out}, O_H, O_W)$

The backward pass reverses this:

```python
        def _backward():
            dout = out.grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

            if self.weight.requires_grad:
                _accum_grad(self.weight, dout.T @ cols)

            if self.bias.requires_grad:
                _accum_grad(self.bias, dout.sum(axis=0))

            if x.requires_grad:
                dcols = dout @ self.weight.data
                dx = col2im(dcols, x.data.shape, kh, kw, self.stride, self.padding)
                _accum_grad(x, dx)
```

Same `grad @ data.T` pattern as the matmul backward - weight gradient is `dout.T @ cols`, input gradient goes through `col2im` to scatter back to spatial format.

The conv backward was the hardest part of the entire project. The im2col data has shape `(N, C, kh, kw, OH, OW)` and needs a specific transpose order before reshaping. I had the axes wrong for three days. The forward pass worked perfectly, gradient checking would fail, and the error message was just "max diff = 847.3" - not helpful. The fix was literally changing `.transpose(0, 4, 5, 1, 2, 3)` to the correct order. I only figured it out by writing out the shape at each step on paper.

### MaxPool2d

Max pooling extracts non-overlapping (or overlapping) patches and takes the maximum of each. The forward pass uses NumPy's **stride tricks** to create a view of all patches without copying data:

```python
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size[0]

    def forward(self, x):
        N, C, H, W = x.data.shape
        kh, kw = self.kernel_size
        s = self.stride
        OH = (H - kh) // s + 1
        OW = (W - kw) // s + 1

        strides = x.data.strides
        patches = np.lib.stride_tricks.as_strided(
            x.data,
            shape=(N, C, OH, OW, kh, kw),
            strides=(strides[0], strides[1], strides[2] * s, strides[3] * s, strides[2], strides[3])
        )
        out_data = patches.max(axis=(4, 5))
```

`as_strided` creates a *view* into the same memory with custom strides; we see the data as patches without copying anything. 

The backward pass needs to route gradients only through the positions that were the maximum:

```python
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                max_vals = out.data[:, :, :, :, None, None]
                mask = (saved_patches == max_vals).astype(np.float32)
                mask_sum = mask.sum(axis=(4, 5), keepdims=True)
                mask = mask / np.maximum(mask_sum, 1.0)
                weighted = mask * out.grad[:, :, :, :, None, None]
                for i in range(kh):
                    for j in range(kw):
                        x.grad[:, :, i:i+s*OH:s, j:j+s*OW:s] += weighted[:, :, :, :, i, j]
```

The mask identifies which input positions were the maximum. If there are ties (multiple positions equal to the max), the gradient is split evenly among them - that's the `mask / mask_sum` line. Then the weighted gradients are scattered back to the input positions using strided indexing.

**Flatten** converts spatial feature maps to vectors for the fully connected layers:

```python
class Flatten(Module):
    def forward(self, x):
        batch_size = x.data.shape[0]
        return x.reshape(batch_size, -1)
```

`reshape` already has a backward closure, so Flatten gets its backward for free.

---

## 10. putting it all together

### simple regression

Learn $y = 2x + 1$:

```python
import numpy as np
from minitorch import Tensor, Sequential, Linear, SGD, mse_loss

np.random.seed(0)
x_np = np.random.rand(100, 1).astype(np.float32)
y_np = 2 * x_np + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

x = Tensor(x_np)
y = Tensor(y_np)

model = Sequential(Linear(1, 1))
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(100):
    pred = model(x)
    loss = mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Forward, loss, zero_grad, backward, step. Five lines, same whether you're fitting a line or training a language model.

### MNIST MLP

```python
def build_mlp():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )
```

784 inputs (28x28 flattened), 128 hidden, 10 classes. Trained with Adam, batch size 100, 15 epochs on 10k samples:

```
Epoch 1/15,  Loss: 0.7578
Epoch 5/15,  Loss: 0.1746
Epoch 10/15, Loss: 0.0780
Epoch 15/15, Loss: 0.0356
Test Accuracy: 95.15%
```

95% on MNIST with a two-layer network and ~1,300 lines of framework code.

### MNIST CNN

```python
def build_cnn():
    return Sequential(
        Conv2d(1, 16, 3, padding=1),   # 28x28 -> 28x28
        ReLU(),
        MaxPool2d(2),                    # 28x28 -> 14x14
        Conv2d(16, 32, 3, padding=1),   # 14x14 -> 14x14
        ReLU(),
        MaxPool2d(2),                    # 14x14 -> 7x7
        Flatten(),                       # 32*7*7 = 1568
        Linear(1568, 128),
        ReLU(),
        Dropout(0.25),
        Linear(128, 10),
    )
```

Two conv-relu-pool blocks, flatten, two linear layers with dropout. 10 epochs on 2k samples:

```
Epoch 1/10,  Loss: 1.0522
Epoch 5/10,  Loss: 0.1272
Epoch 10/10, Loss: 0.0398
Test Accuracy: 96.40%
```

96.4% with a CNN on only 2,000 training samples. The convolutions are doing real work - same accuracy the MLP needs 10k samples to reach.

The training loop is the standard five-line pattern, and evaluation uses `no_grad()` to skip graph building. The API is intentionally PyTorch-compatible - if you know one, you know the other.

---

## 11. gradient checking

The only reliable check: compute gradients the dumb way via finite differences and compare:

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Central difference, $O(\epsilon^2)$ accurate.

```python
def numerical_gradient(f, inputs, eps=1e-5):
    grads = []
    orig_data = [inp.data.copy() for inp in inputs]
    for inp in inputs:
        inp.data = inp.data.astype(np.float64)
    f64_data = [inp.data.copy() for inp in inputs]

    for k, inp in enumerate(inputs):
        grad = np.zeros(inp.data.shape, dtype=np.float64)
        it = np.nditer(f64_data[k], flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = f64_data[k][idx]

            inp.data = f64_data[k].copy()
            inp.data[idx] = old_val + eps
            loss_plus = float(f().data)

            inp.data = f64_data[k].copy()
            inp.data[idx] = old_val - eps
            loss_minus = float(f().data)

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        grads.append(grad.astype(np.float32))
    return grads
```

Note the conversion to float64. Finite differences with float32 and small epsilon can give garbage results due to catastrophic cancellation. Float64 gives us enough precision.

The `check_gradient` function runs both the analytical backward pass and the numerical approximation, then compares with `np.allclose`:

```python
def check_gradient(f, inputs, eps=1e-5, atol=1e-4, rtol=1e-3):
    for inp in inputs:
        inp.zero_grad()
    loss = f()
    loss.backward()
    analytic = [inp.grad.copy() for inp in inputs]

    numerical = numerical_gradient(f, inputs, eps)

    for i, (a, n) in enumerate(zip(analytic, numerical)):
        if not np.allclose(a, n, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(a - n))
            raise AssertionError(
                f"Gradient check failed for input {i}: max diff = {max_diff}\n"
                f"Analytic:\n{a}\nNumerical:\n{n}"
            )
    return True
```

Slow (evaluates `f` twice per parameter element), but it caught every backward bug I introduced. Silently wrong gradients are the worst kind of bug in deep learning - your model trains, loss goes down, accuracy is just... bad. Gradient checking is the only reliable way to catch this. I ran it on every single operation before trusting any backward implementation.

---

## 12. visualization

Renders the computation graph with graphviz:

```python
def draw_graph(root, format='svg'):
    from graphviz import Digraph
    dot = Digraph(format=format, graph_attr={'rankdir': 'LR'})

    visited = set()
    def visit(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        uid = str(id(v))

        if v.data.ndim == 0:
            label = f'{float(v.data):.3g}'
        else:
            label = str(tuple(v.shape))

        color = '#d4edda' if v.requires_grad else '#e2e3e5'
        dot.node(uid, label, shape='ellipse', style='filled', fillcolor=color)

        if v._op:
            op_uid = uid + '_op'
            dot.node(op_uid, v._op, shape='box', style='filled', fillcolor='#cce5ff')
            dot.edge(op_uid, uid)
            for child in v._prev:
                visit(child)
                dot.edge(str(id(child)), op_uid)

    visit(root)
    return dot
```

Green ellipses are tensors with `requires_grad=True`, gray for constants, blue boxes for operations. When your gradients are wrong, visualize the graph first - a missing edge or detached tensor is immediately obvious in the picture.

---

## 13. what's missing (and how hard it would be)

minitorch gets 96% on MNIST. Real PyTorch trains GPT-4. Here's the gap:

**Compiled kernels** - the big one. Everything here runs through interpreted Python and NumPy. A production framework compiles to C++/CUDA and fuses operations into single kernel launches. Our im2col Conv2d is maybe 100-1000x slower than cuDNN. Closing this gap is a year of work minimum, also I do not CUDA (yet; I'm [learning](https://www.manning.com/books/cuda-for-deep-learning))

**Real GPU support** - there's an optional CuPy backend, but im2col still runs on CPU. A real CUDA implementation would need custom kernels.

**In-place operations** - every op creates a new tensor. This simplifies the graph (no aliasing) but wastes memory. Supporting `x.add_(y)` requires version counting to detect when in-place ops invalidate the graph. Which I was unable to do.

**Distributed training, mixed precision, JIT** - each of these is its own project. Distributed needs NCCL and gradient all-reduce. Mixed precision needs loss scaling. JIT needs a tracer and compiler.

There's no magic anywhere; just well-organized linear algebra and the chain rule, all the way down.

---

## 14. references

- [neural networks from scratch](/tech/2025/10/25/nn.html) - building a neural network without a framework
- [tensors and autograd](/tech/2026/01/07/torch.html) - the math behind automatic differentiation
- [backpropagation math](/tech/2026/01/21/math.html) - deriving the chain rule for neural networks
- [micrograd](https://github.com/karpathy/micrograd) - Karpathy's scalar-valued autograd engine
- [tinygrad](https://github.com/tinygrad/tinygrad) - George Hotz's minimal deep learning framework
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) - Edward Yang's tour of PyTorch's C++ core
- [Kaiming He et al., "Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852) - the Kaiming initialization paper
- [Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks"](https://proceedings.mlr.press/v9/glorot10a.html) - the Xavier initialization paper
- [Kingma & Ba, "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - the Adam optimizer paper
- [Ioffe & Szegedy, "Batch Normalization"](https://arxiv.org/abs/1502.03167) - the batch normalization paper
