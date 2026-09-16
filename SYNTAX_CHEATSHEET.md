# PyTorch Syntax Cheat Sheet

You don't need to memorize any of this — just use it as a reference while you fill
in the `#TODO`s. Every pattern used anywhere in the bootcamp is listed here, with a
plain-English explanation and a generic example you can adapt.

---

### Calling a layer or model on some input

A layer (like `nn.Linear`) or a full model works like a function: you call it with
parentheses and give it your input.

```python
output = some_layer(some_input)
```

If a model defines a layer in `__init__` (e.g. `self.some_name = nn.Linear(...)`),
you call it the same way inside `forward`: `self.some_name(x)`.

---

### Computing a loss

A loss function (like `nn.MSELoss()`) also gets called like a function, with your
model's prediction and the real answer as arguments.

```python
loss = loss_function(prediction, target)
```

---

### Backward pass (computing gradients)

Once you have a `loss` tensor, calling `.backward()` on it computes how much each
model parameter contributed to that loss. You don't pass it any arguments.

```python
loss.backward()
```

---

### Optimizer step (updating the model)

After `.backward()` has computed gradients, the optimizer's `.step()` method
actually applies them — nudging the model's parameters to reduce the loss.

```python
optimizer.step()
```

Always call `optimizer.zero_grad()` before the next `.backward()`, or gradients
will pile up from previous steps instead of being recalculated fresh.

---

### Sampling an action from a probability distribution

`torch.distributions.Categorical` wraps a set of probabilities (like `[0.7, 0.3]`)
into an object you can *sample* from — like a loaded die. `.sample()` picks one
option randomly, weighted by those probabilities. `.log_prob(x)` tells you how
"surprising" a specific outcome `x` was (used later to compute the loss).

```python
distribution = torch.distributions.Categorical(some_probabilities)
choice = distribution.sample()               # pick one, e.g. tensor(1)
how_likely = distribution.log_prob(choice)    # log-probability of that choice
```

---

### Looping backwards through a list

`reversed(range(len(some_list)))` gives you the valid indices of `some_list`,
counted from the last one down to 0. Useful when a later value depends on values
that come after it (like discounted returns, computed from the end of an episode
backwards).

```python
for i in reversed(range(len(some_list))):
    ...  # i goes: last index, ..., 1, 0
```

---

### Building a tensor from a list, elementwise

`zip(list_a, list_b)` pairs up two lists element-by-element. Combined with a list
comprehension and `torch.stack`, this is how you build one loss value out of many
per-step values:

```python
combined = torch.stack([f(a, b) for a, b in zip(list_a, list_b)])
total = combined.sum()
```

---

### Quick reference table

| Syntax | What it does |
| --- | --- |
| `layer(x)` | run input `x` through a layer/model |
| `loss_fn(pred, target)` | compute how wrong `pred` is |
| `loss.backward()` | compute gradients |
| `optimizer.step()` | apply gradients to update the model |
| `optimizer.zero_grad()` | clear old gradients before the next backward pass |
| `Categorical(probs).sample()` | randomly pick an action, weighted by `probs` |
| `dist.log_prob(x)` | how likely outcome `x` was under `dist` |
| `reversed(range(len(x)))` | loop through indices of `x` back-to-front |
