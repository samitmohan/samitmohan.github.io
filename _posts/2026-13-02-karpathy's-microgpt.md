---
layout: post
title: "Karpathy’s MicroGPT"
date: 2026-02-13 14:06:04 +0530
categories: tech
---

Explaining the algorithm behind Transformers; the reason why your job will be gone; in 200 lines.

---

## 1. The Autograd Engine: `Value` Class

At the heart of the script is the `Value` class, a scalar-based automatic differentiation engine. 

```python
class Value:
    def __init__(self, data, _children=(), local_grads = ()):
        self.data = data
        self.grad = 0
        self._children = _children
        self.local_grads = local_grads
```

Every operation (addition, multiplication, etc.) creates a new `Value` object that remembers its "parents" (`_children`) and its "local gradient" (the derivative of the operation). When `backward()` is called, it performs a topological sort and applies the **chain rule** to propagate gradients from the loss back to every single parameter in the model.

## 2. The Transformer Architecture

The model follows the GPT-2/LLaMA style architecture:

-   **Embeddings**: Token embeddings (`wte`) and Position embeddings (`wpe`) are added together.
-   **RMSNorm**: A modern variation of LayerNorm that normalizes the activations based on their root mean square.
-   **Multi-Head Attention (MHA)**: This is where the magic happens. 
    -   It computes **Queries (Q)**, **Keys (K)**, and **Values (V)** for each head.
    -   **Attention Scores** are calculated by the dot product of Q and K, divided by the square root of the head dimension.
    -   **Causal Masking** is implicit in this implementation because it appends new keys/values to a cache during the sequence.
-   **MLP (Feed-Forward)**: A simple two-layer linear network with a ReLU activation in between.
-   **Residual Connections**: `x = x_attention + x_residual` ensures that gradients can flow easily through deep networks.

## 3. The Training Loop
The script trains on a list of names to predict the next character.

```python
for pos_id in range(n):
    logits = gpt(token_id, pos_id, keys, values)
    probs = softmax(logits)
    loss_target = -probs[target_id].log() # Cross-Entropy Loss
    losses.append(loss_target)
```

It uses the **Adam Optimizer**, implemented manually. Adam tracks the first and second moments of the gradients to adaptively change the learning rate for each parameter, ensuring faster and more stable convergence than simple SGD.

## 4. Inference and Sampling
After training, the model generates new names by sampling from the output distribution:

```python
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

The **Temperature** parameter controls creativity: lower temperature makes the model more confident (picking only the most likely characters), while higher temperature introduces more randomness.

## Conclusion

`microgpt.py` is a masterclass in minimalism. It proves that the "intelligence" of large language models isn't magic—it's just a massive DAG of additions and multiplications, meticulously tracked by calculus.

## Code:

```python
import os 
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

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_logit = max(l.data for l in logits)
    exp_logits = [(logit - max_logit).exp() for logit in logits] # subtract max logit for numerical stability
    sum_exp_logits = sum(exp_logits)
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

def rmsnorm(x):
    eps = 1e-8
    mean_square = sum(xi**2 for xi in x) / len(x)
    root_mean_square = (mean_square + eps)**0.5
    return [xi / root_mean_square for xi in x]

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

num_steps = 20 # training steps
for step in range(num_steps):
    # forward pass
    # take a document from the dataset and convert it to a list of token IDs (add BOS token at the beginning)
    total_loss = Value(0.0)

    batch_size = 8
    for b in range(batch_size):
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
    # backward pass
    loss.backward() # compute the gradients of the loss with respect to all parameters in the computational

    # update the parameters using Adam
    learning_rate_target = learning_rate * (1 - step / num_steps) # linearly decay the learning rate to zero over the course of training
    for i, param in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * param.grad # update the first moment vector for the current parameter
        v[i] = beta2 * v[i] + (1 - beta2) * param.grad**2 # update the second moment vector for the current parameter
        m_hat = m[i] / (1 - beta1**(step + 1)) # compute the bias-corrected first moment estimate
        v_hat = v[i] / (1 - beta2**(step + 1)) # compute the bias-corrected second moment estimate
        param.data -= learning_rate_target * m_hat / (math.sqrt(v_hat) + eps_adam) # update the parameter using the Adam update rule
        param.grad = 0 # reset the gradient of the parameter to zero for the next iteration

    print(f"Step {step + 1}/{num_steps}, Loss: {loss.data:.4f}")

# inference
temperature = 0.5 # control creativity of generated text
print("\nGenerated names:")
for _ in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)] # reset the list of key and value vectors for each transformer block
    token_id = BOS # start with the BOS token
    generated_name = 'har'
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