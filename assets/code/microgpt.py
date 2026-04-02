# microgpt.py
# A minimal GPT implementation in pure Python, no frameworks.
# Every list comprehension is written as an explicit for-loop so you can
# read each step without knowing Python tricks.
#
# Blog post: https://samitmohan.github.io
# Reference dataset: https://github.com/karpathy/makemore

# ============================================================
# SECTION 0: imports
# ============================================================
import math
import os
import random
import urllib.request

# ============================================================
# SECTION 1: data loading
# ============================================================
url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
if not os.path.exists("names.txt"):
    urllib.request.urlretrieve(url, "names.txt")

docs = open("names.txt").read().splitlines()
print(f"num docs: {len(docs)}")

# ============================================================
# SECTION 2: vocabulary setup
# ============================================================
# Build a sorted list of every unique character that appears in the names.
all_chars = set()
for doc in docs:
    for c in doc:
        all_chars.add(c)
chars = sorted(all_chars)   # ['a', 'b', ..., 'z'] - 26 characters

BOS = len(chars)            # integer ID 26: beginning-of-sequence token
vocab_size = len(chars) + 1 # 27 tokens total
print(f"vocab size: {vocab_size}")

# BOS doubles as EOS: every name is wrapped as [BOS, c1, c2, ..., cn, BOS].
# When the model emits BOS during generation that is the stop signal.

# ============================================================
# SECTION 3: autograd engine
# ============================================================
# Every scalar in the model is wrapped in a Value node.
# Each node records:
#   data       - the numeric value
#   grad       - dLoss/d(this), accumulated by backward()
#   _children  - Value nodes that produced this one
#   local_grads - d(this)/d(each child), same order as _children
#
# backward() walks the graph in reverse and applies the chain rule,
# filling in grad for every node in one pass.

class Value:
    def __init__(self, data, _children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = _children
        self.local_grads = local_grads

    # d(a+b)/da = 1,  d(a+b)/db = 1
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other))
        out.local_grads = (1.0, 1.0)
        return out

    # d(a*b)/da = b,  d(a*b)/db = a
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other))
        out.local_grads = (other.data, self.data)
        return out

    # d(x^n)/dx = n * x^(n-1)
    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    # d(log x)/dx = 1/x   (small epsilon avoids log(0))
    def log(self):
        return Value(math.log(self.data + 1e-10), (self,), (1 / (self.data + 1e-10),))

    # d(exp x)/dx = exp(x)
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # d(relu x)/dx = 1 if x > 0, else 0
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # Convenience ops - they reduce to the three primitives above,
    # so no new gradient rules are needed.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    # Reflected operators: Python calls these when the left operand cannot handle
    # the operation.  Example: 3 * some_value calls int.__mul__ first, which fails,
    # then Value.__rmul__ is tried.  Without these, plain numbers on the left would crash.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        # Step 1: build a topological order of the computation graph using DFS.
        # Post-order means a node appears AFTER all the nodes it depends on.
        topological_graph = []
        visited = set()

        def build_top(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_top(child)
                topological_graph.append(v)   # added AFTER all children

        build_top(self)

        # Step 2: seed the gradient at the root (dLoss/dLoss = 1) and propagate.
        self.grad = 1.0
        for v in reversed(topological_graph):   # root first, leaves last
            for child, local_grad in zip(v._children, v.local_grads):
                child.grad += local_grad * v.grad   # chain rule: dLoss/dchild += local * dLoss/dv

# ============================================================
# SECTION 4: hyperparameters
# ============================================================
n_embed    = 16   # every vector in the model is this long
n_head     = 4    # attention heads (each uses head_dim = 4 dimensions)
n_layer    = 1    # number of transformer blocks stacked
block_size = 16   # maximum sequence length
head_dim   = n_embed // n_head   # = 4

# ============================================================
# SECTION 5: weight initialization
# ============================================================

def matrix(rows, cols):
    # Allocate a 2D list of Value nodes with small random initial values.
    # Small sigma (0.08) prevents activations from exploding at startup.
    result = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(Value(random.gauss(0, 0.08)))
        result.append(row)
    return result


# Token embedding table: one learned 16-dim vector per vocabulary token.
# The model looks up wte[token_id] to get the embedding for that token.
wte = matrix(vocab_size, n_embed)   # 27 x 16 = 432 params

# Position embedding table: one learned 16-dim vector per position index.
# Added to the token embedding so the model knows where each token sits.
wpe = matrix(block_size, n_embed)   # 16 x 16 = 256 params

# Per-layer attention and MLP weights.
# These are lists so that each transformer block has its own independent copy.
wq   = []   # query projection matrices
wk   = []   # key projection matrices
wv   = []   # value projection matrices
wo   = []   # output projection (mixes the concatenated head outputs)
w_fc1 = []  # MLP first layer:  n_embed -> n_embed*4
w_fc2 = []  # MLP second layer: n_embed*4 -> n_embed

for i in range(n_layer):
    wq.append(matrix(n_embed, n_embed))          # 16 x 16 = 256 params
    wk.append(matrix(n_embed, n_embed))          # 16 x 16 = 256 params
    wv.append(matrix(n_embed, n_embed))          # 16 x 16 = 256 params
    wo.append(matrix(n_embed, n_embed))          # 16 x 16 = 256 params
    w_fc1.append(matrix(n_embed * 4, n_embed))   # 64 x 16 = 1024 params
    w_fc2.append(matrix(n_embed, n_embed * 4))   # 16 x 64 = 1024 params

# Language model head: projects each 16-dim output vector to 27 logits
# (one logit per vocabulary token).
lm_head = matrix(vocab_size, n_embed)   # 27 x 16 = 432 params

# Collect every Value object into a flat list so the SGD loop can iterate them.
def flatten(mat):
    result = []
    for row in mat:
        for val in row:
            result.append(val)
    return result

params = flatten(wte) + flatten(wpe)
for i in range(n_layer):
    params += flatten(wq[i]) + flatten(wk[i]) + flatten(wv[i]) + flatten(wo[i])
    params += flatten(w_fc1[i]) + flatten(w_fc2[i])
params += flatten(lm_head)

print(f"num params: {len(params)}")
# Expected: 432 + 256 + 1*(256*4 + 1024 + 1024) + 432 = 4192

# ============================================================
# SECTION 6: utility functions
# ============================================================

def linear(x, w):
    # Matrix-vector multiply: output[i] = dot(w[i], x).
    # Each row of w contains the weights for one output dimension.
    result = []
    for row in w:
        total = Value(0.0)
        for weight, inp in zip(row, x):
            total = total + weight * inp
        result.append(total)
    return result


def softmax(logits):
    # Convert raw scores to a probability distribution.
    # Numerically stable version: subtract the max before exponentiating.
    # The constant cancels in the ratio, so the result is identical to naive softmax.
    max_logit = max(l.data for l in logits)

    exp_logits = []
    for logit in logits:
        exp_logits.append((logit - max_logit).exp())

    sum_exp = Value(0.0)
    for e in exp_logits:
        sum_exp = sum_exp + e

    result = []
    for e in exp_logits:
        result.append(e / sum_exp)
    return result


def rmsnorm(x):
    # Scale x so that its root-mean-square magnitude is approximately 1.
    # Keeps activations in a sane range throughout training.
    eps = 1e-8
    sq_total = Value(0.0)
    for xi in x:
        sq_total = sq_total + xi ** 2
    mean_sq = sq_total / len(x)
    rms = (mean_sq + eps) ** 0.5
    result = []
    for xi in x:
        result.append(xi / rms)
    return result

# ============================================================
# SECTION 7: model components
# ============================================================

def attention_block(x_seq, wq_i, wk_i, wv_i, wo_i):
    # Project every token embedding to Query, Key, and Value vectors.
    # Q: "what am I looking for?"
    # K: "what do I advertise?"
    # V: "what information do I carry?"
    Q = []
    for x in x_seq:
        Q.append(linear(x, wq_i))
    K = []
    for x in x_seq:
        K.append(linear(x, wk_i))
    V = []
    for x in x_seq:
        V.append(linear(x, wv_i))

    out_seq = []
    for t in range(len(x_seq)):
        # x_attn collects the concatenated head outputs for position t
        x_attn = []

        for h in range(n_head):
            start = h * head_dim
            end   = (h + 1) * head_dim

            # Query slice for this head at the current position
            qh = Q[t][start:end]

            # Compute attention scores against all past (and current) positions.
            # Using range(t+1) enforces causal masking: future tokens are never seen.
            scores = []
            for s in range(t + 1):
                score = Value(0.0)
                for qi, ki in zip(qh, K[s][start:end]):
                    score = score + qi * ki
                # Scale by 1/sqrt(head_dim) to prevent dot products from growing too large.
                # Without scaling, softmax saturates and gradients vanish.
                score = score / math.sqrt(head_dim)
                scores.append(score)

            # Turn scores into weights that sum to 1
            weights = softmax(scores)

            # Weighted average of value vectors across attended positions
            head_out = []
            for d in range(head_dim):
                val = Value(0.0)
                for s in range(t + 1):
                    val = val + weights[s] * V[s][start + d]
                head_out.append(val)

            # Concatenate this head's output onto x_attn
            for elem in head_out:
                x_attn.append(elem)

        # Final linear projection mixes information across the concatenated heads
        out_seq.append(linear(x_attn, wo_i))

    return out_seq


def mlp_block(x, w_fc1_i, w_fc2_i):
    # Two-layer feed-forward network applied independently at each position.
    # Pattern: expand -> non-linearity -> contract
    x = linear(x, w_fc1_i)   # 16 -> 64: project to wider space

    relu_out = []
    for xi in x:
        relu_out.append(xi.relu())   # zero out negatives; introduces non-linearity

    x = linear(relu_out, w_fc2_i)   # 64 -> 16: project back to model dimension
    return x

# ============================================================
# SECTION 8: full forward pass
# ============================================================

def gpt(tokens):
    # Step 1: turn integer token IDs into continuous vectors.
    # Each position gets: wte[token_id] + wpe[position] (element-wise add).
    # Without position embeddings the model cannot distinguish order.
    x_seq = []
    for pos, tok in enumerate(tokens):
        emb = []
        for te, pe in zip(wte[tok], wpe[pos]):
            emb.append(te + pe)
        x_seq.append(emb)

    # Step 2: run each transformer block in sequence.
    # Every block has two sub-layers, each with pre-normalization and a residual connection.
    #   Attention sub-layer: lets tokens read from each other
    #   MLP sub-layer: per-token computation (no cross-token interaction)
    for i in range(n_layer):

        # --- attention sub-layer ---
        normed = []
        for x in x_seq:
            normed.append(rmsnorm(x))

        attn_out = attention_block(normed, wq[i], wk[i], wv[i], wo[i])

        # Residual connection: add attention output back to the pre-attention values.
        # This gives gradients a direct path back through the network (no vanishing).
        new_x_seq = []
        for attn, x in zip(attn_out, x_seq):
            combined = []
            for a, b in zip(attn, x):
                combined.append(a + b)
            new_x_seq.append(combined)
        x_seq = new_x_seq

        # --- MLP sub-layer ---
        new_x_seq = []
        for x in x_seq:
            mlp_out = mlp_block(rmsnorm(x), w_fc1[i], w_fc2[i])
            combined = []
            for a, b in zip(mlp_out, x):
                combined.append(a + b)
            new_x_seq.append(combined)
        x_seq = new_x_seq

    # Step 3: project each 16-dim vector to vocab_size logits.
    # logits_seq[t][v] = unnormalized score that position t predicts token v next.
    logits_seq = []
    for x in x_seq:
        logits_seq.append(linear(x, lm_head))

    return logits_seq

# ============================================================
# SECTION 9: training loop
# ============================================================
num_steps     = 20    # increase to 500+ for much better names
batch_size    = 8
learning_rate = 0.1

for step in range(num_steps):
    total_loss = Value(0.0)

    for b in range(batch_size):
        # Cycle through the dataset repeatedly using modulo
        doc = docs[(step * batch_size + b) % len(docs)]

        # Tokenize: convert each character to its integer index in chars
        char_tokens = []
        for c in doc:
            char_tokens.append(chars.index(c))
        tokens = [BOS] + char_tokens + [BOS]
        tokens = tokens[:block_size + 1]   # truncate names longer than block_size

        input_tokens  = tokens[:-1]   # context the model sees
        target_tokens = tokens[1:]    # next token the model must predict at each position

        logits_seq = gpt(input_tokens)

        # Cross-entropy loss: -log(probability assigned to the correct token)
        # Random baseline with 27 classes: -log(1/27) ~= 3.3
        loss = Value(0.0)
        for logits, target in zip(logits_seq, target_tokens):
            probs = softmax(logits)
            loss = loss + (-probs[target].log())   # penalize low probability on the right token
        total_loss = total_loss + loss / len(logits_seq)

    avg_loss = total_loss / batch_size
    avg_loss.backward()   # fills .grad for all 4192 parameters in one reverse pass

    # SGD update: nudge each parameter in the direction that reduces loss
    for param in params:
        param.data -= learning_rate * param.grad
        param.grad = 0.0   # IMPORTANT: reset before next step or gradients accumulate

    print(f"step {step + 1:3d}/{num_steps} | loss {avg_loss.data:.4f}")

# ============================================================
# SECTION 10: inference
# ============================================================
# Generate names one token at a time.
# Seed the context with [BOS], run the model, sample the next token,
# append it, repeat until the model outputs BOS again (end-of-name signal).

temperature = 0.5   # lower = more predictable/repetitive, higher = more creative/random

print("\nGenerated names:")
for _ in range(10):
    context = [BOS]
    name = ""

    for _ in range(block_size):
        logits_seq = gpt(context)

        # Apply temperature: divide logits before softmax.
        # Lower T sharpens the distribution (model picks safer tokens).
        # Higher T flattens it (model takes more risks).
        logits_t = []
        for l in logits_seq[-1]:
            logits_t.append(l / temperature)

        probs = softmax(logits_t)

        # Sample one token according to its probability
        prob_data = []
        for p in probs:
            prob_data.append(p.data)
        next_token = random.choices(range(vocab_size), weights=prob_data)[0]

        if next_token == BOS:   # model signaled end of name
            break

        context.append(next_token)
        name += chars[next_token]

    print(name)
