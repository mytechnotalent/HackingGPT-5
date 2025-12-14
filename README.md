![image](https://github.com/mytechnotalent/HackingGPT-5/blob/main/HackingGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# HackingGPT
## Part 5
Part 5 covers softmax-based token averaging with masking, the negative infinity trick for masking future positions, understanding how e^(-inf) = 0 enables causal attention, and comparing softmax weights to manual normalization.

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)

<br>

## Part 4 [HERE](https://github.com/mytechnotalent/HackingGPT-4)

<br><br>

```python
import torch
from torch.nn import functional as F
```


## Step 1: Load and Inspect the Data
Now let's read the file and see what we're working with. Understanding your data is crucial before building any model!


```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```


```python
text
```


**Output:**
```
'A dim glow rises behind the glass of a screen and the machine exhales in binary tides. The hum is a language and one who listens leans close to catch the quiet grammar. Patterns fold like small maps and seams hint at how the thing holds itself together. Treat each blinking diode and each idle tick as a sentence in a story that asks to be read.\n\nThere is patience here, not of haste but of careful unthreading. Where others see a sealed box the curious hand traces the join and wonders which thought made it fit. Do not rush to break, coax the meaning out with questions, and watch how the logic replies in traces and errors and in the echoes of forgotten interfaces.\n\nTechnology is artifact and argument at once. It makes a claim about what should be simple, what should be hidden, and what should be trusted. Reverse the gaze and learn its rhetoric, see where it promises ease, where it buries complexity, and where it leaves a backdoor as a sigh between bricks. To read that rhetoric is to be a kind interpreter, not a vandal.\n\nThis work is an apprenticeship in humility. Expect bafflement and expect to be corrected by small things, a timing oddity, a mismatch of expectation, a choice that favors speed over grace. Each misstep teaches a vocabulary of trade offs. Each discovery is a map of decisions and not a verdict on worth.\n\nThere is a moral keeping in the craft. Let curiosity be tempered with regard for consequence. Let repair and understanding lead rather than exploitation. The skill that opens a lock should also know when to hold the key and when to hand it back, mindful of harm and mindful of help.\n\nCelebrate the quiet victories, a stubborn protocol understood, an obscure format rendered speakable, a closed device coaxed into cooperation. These are small reconciliations between human intent and metal will, acts of translation rather than acts of conquest.\n\nAfter decoding a mechanism pause and ask what should change, a bug to be fixed, a user to be warned, a design to be amended. The true maker of machines leaves things better for having looked, not simply for having cracked the shell.'
```


## Step 2: Version 3 - Using Softmax and Masking
In real attention, we use **softmax** to create the weights. This requires a trick where we use `-inf` to mask out future positions (because e^(-inf) = 0).

### Why Use Softmax Instead of Manual Division?
In Part 4, we normalized by dividing each row by its sum. This works for uniform averaging, but in real transformers the following happens.
1. Weights are LEARNED, not uniform.
2. Different tokens get different attention weights.
3. Softmax naturally converts any values to probabilities that sum to 1.

### The Softmax Function
Softmax takes a vector of any real numbers and converts them to probabilities.
```
softmax([x1, x2, x3]) = [e^x1, e^x2, e^x3] / (e^x1 + e^x2 + e^x3)
```

Properties of softmax are as follows.
1. All outputs are positive (because e^x > 0 for any x).
2. All outputs sum to 1 (because we divide by total).
3. Larger inputs get larger probabilities.
4. e^(-inf) = 0, so -inf inputs become 0 probability.

### The Masking Trick
To prevent looking at future tokens, we set future positions to -inf BEFORE applying softmax.
| Position | Raw weights | After masking | After softmax |
|----------|-------------|---------------|---------------|
| row 0 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, -inf, -inf, -inf, -inf, -inf, -inf, -inf] | [1.0, 0, 0, 0, 0, 0, 0, 0] |
| row 1 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, -inf, -inf, -inf, -inf, -inf, -inf] | [0.5, 0.5, 0, 0, 0, 0, 0, 0] |
| row 2 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, -inf, -inf, -inf, -inf, -inf] | [0.33, 0.33, 0.33, 0, 0, 0, 0, 0] |
| row 3 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, -inf, -inf, -inf, -inf] | [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0] |
| row 4 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, -inf, -inf, -inf] | [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0] |
| row 5 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, -inf, -inf] | [0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0, 0] |
| row 6 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0, -inf] | [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0] |
| row 7 | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0, 0] | [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125] |


```python
torch.manual_seed(42)
```


**Output:**
```
<torch._C.Generator at 0x10ab4e4f0>
```


```python
# define batch dimension
B = 4  # batch size: 4 independent sequences
B
```


**Output:**
```
4
```


```python
# define time dimension
T = 8  # sequence length: 8 tokens/positions in each sequence
T
```


**Output:**
```
8
```


```python
# define channel dimension
C = 2  # feature size: 2 features per token
C
```


**Output:**
```
2
```


```python
# start with random data
x = torch.randn(B, T, C)
x
```


**Output:**
```
tensor([[[ 1.9269,  1.4873],
         [ 0.9007, -2.1055],
         [ 0.6784, -1.2345],
         [-0.0431, -1.6047],
         [-0.7521,  1.6487],
         [-0.3925, -1.4036],
         [-0.7279, -0.5594],
         [-0.7688,  0.7624]],

        [[ 1.6423, -0.1596],
         [-0.4974,  0.4396],
         [-0.7581,  1.0783],
         [ 0.8008,  1.6806],
         [ 1.2791,  1.2964],
         [ 0.6105,  1.3347],
         [-0.2316,  0.0418],
         [-0.2516,  0.8599]],

        [[-1.3847, -0.8712],
         [-0.2234,  1.7174],
         [ 0.3189, -0.4245],
         [ 0.3057, -0.7746],
         [-1.5576,  0.9956],
         [-0.8798, -0.6011],
         [-1.2742,  2.1228],
         [-1.2347, -0.4879]],

        [[-0.9138, -0.6581],
         [ 0.0780,  0.5258],
         [-0.4880,  1.1914],
         [-0.8140, -0.7360],
         [-1.4032,  0.0360],
         [-0.0635,  0.6756],
         [-0.0978,  1.8446],
         [-1.1845,  1.3835]]])
```


```python
# create the mask (lower triangular)
# this is the same lower triangular matrix from Part 4
# 1s where we CAN look, 0s where we CANNOT look
tril = torch.tril(torch.ones(T, T))
tril
```


**Output:**
```
tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1.]])
```


```python
# examine each row of tril to understand the mask
print('examining each row of the lower triangular mask')
print()
print(f'row 0: {tril[0].tolist()}')
print('       position 0 can only see itself')
print('       1 = can look, 0 = cannot look')
print()
print(f'row 1: {tril[1].tolist()}')
print('       position 1 can see positions 0 and 1')
print()
print(f'row 2: {tril[2].tolist()}')
print('       position 2 can see positions 0, 1, and 2')
print()
print(f'row 3: {tril[3].tolist()}')
print('       position 3 can see positions 0, 1, 2, and 3')
print()
print(f'row 4: {tril[4].tolist()}')
print('       position 4 can see positions 0, 1, 2, 3, and 4')
print()
print(f'row 5: {tril[5].tolist()}')
print('       position 5 can see positions 0, 1, 2, 3, 4, and 5')
print()
print(f'row 6: {tril[6].tolist()}')
print('       position 6 can see positions 0, 1, 2, 3, 4, 5, and 6')
print()
print(f'row 7: {tril[7].tolist()}')
print('       position 7 can see all positions 0, 1, 2, 3, 4, 5, 6, and 7')
```


**Output:**
```
examining each row of the lower triangular mask

row 0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 0 can only see itself
       1 = can look, 0 = cannot look

row 1: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 1 can see positions 0 and 1

row 2: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       position 2 can see positions 0, 1, and 2

row 3: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
       position 3 can see positions 0, 1, 2, and 3

row 4: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
       position 4 can see positions 0, 1, 2, 3, and 4

row 5: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
       position 5 can see positions 0, 1, 2, 3, 4, and 5

row 6: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
       position 6 can see positions 0, 1, 2, 3, 4, 5, and 6

row 7: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
       position 7 can see all positions 0, 1, 2, 3, 4, 5, 6, and 7

```


```python
# start with zeros (equal weights before softmax)
# in real attention, these would be LEARNED values
# for now, we use zeros to show the mechanism
wei = torch.zeros((T, T))
wei
```


**Output:**
```
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```


```python
# understand why we start with zeros
print('understanding the initial weights')
print()
print('wei is an 8x8 matrix of zeros')
print('this represents: every position has EQUAL affinity for every other position')
print()
print('why zeros?')
print('   when we apply softmax to equal values, we get equal probabilities')
print('   softmax([0, 0, 0]) = [0.333, 0.333, 0.333]')
print('   softmax([0, 0]) = [0.5, 0.5]')
print('   softmax([0]) = [1.0]')
print()
print('in real transformers, these values would be LEARNED')
print('different values would give different attention patterns')
```


**Output:**
```
understanding the initial weights

wei is an 8x8 matrix of zeros
this represents: every position has EQUAL affinity for every other position

why zeros?
   when we apply softmax to equal values, we get equal probabilities
   softmax([0, 0, 0]) = [0.333, 0.333, 0.333]
   softmax([0, 0]) = [0.5, 0.5]
   softmax([0]) = [1.0]

in real transformers, these values would be LEARNED
different values would give different attention patterns

```


```python
# set future positions to -infinity
# masked_fill: where tril==0, fill with -inf
# this prevents looking at future tokens
wei = wei.masked_fill(tril == 0, float('-inf'))
wei
```


**Output:**
```
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```


```python
# understand the masked_fill operation step by step
print('understanding masked_fill')
print()
print('tril == 0 creates a boolean mask')
print('True where tril is 0 (future positions)')
print('False where tril is 1 (past/current positions)')
print()
mask = tril == 0
print('tril == 0:')
print(mask)
print()
print('masked_fill replaces values where the mask is True with -inf')
```


**Output:**
```
understanding masked_fill

tril == 0 creates a boolean mask
True where tril is 0 (future positions)
False where tril is 1 (past/current positions)

tril == 0:
tensor([[False,  True,  True,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False, False,  True],
        [False, False, False, False, False, False, False, False]])

masked_fill replaces values where the mask is True with -inf

```


```python
# examine each row of wei after masking
print('examining each row after masking with -inf')
print()
print(f'row 0: {wei[0].tolist()}')
print('       only position 0 is visible (0), positions 1-7 are masked (-inf)')
print()
print(f'row 1: {wei[1].tolist()}')
print('       positions 0-1 are visible (0), positions 2-7 are masked (-inf)')
print()
print(f'row 2: {wei[2].tolist()}')
print('       positions 0-2 are visible (0), positions 3-7 are masked (-inf)')
print()
print(f'row 3: {wei[3].tolist()}')
print('       positions 0-3 are visible (0), positions 4-7 are masked (-inf)')
print()
print(f'row 4: {wei[4].tolist()}')
print('       positions 0-4 are visible (0), positions 5-7 are masked (-inf)')
print()
print(f'row 5: {wei[5].tolist()}')
print('       positions 0-5 are visible (0), positions 6-7 are masked (-inf)')
print()
print(f'row 6: {wei[6].tolist()}')
print('       positions 0-6 are visible (0), position 7 is masked (-inf)')
print()
print(f'row 7: {wei[7].tolist()}')
print('       all positions 0-7 are visible (0), nothing is masked')
```


**Output:**
```
examining each row after masking with -inf

row 0: [0.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
       only position 0 is visible (0), positions 1-7 are masked (-inf)

row 1: [0.0, 0.0, -inf, -inf, -inf, -inf, -inf, -inf]
       positions 0-1 are visible (0), positions 2-7 are masked (-inf)

row 2: [0.0, 0.0, 0.0, -inf, -inf, -inf, -inf, -inf]
       positions 0-2 are visible (0), positions 3-7 are masked (-inf)

row 3: [0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf, -inf]
       positions 0-3 are visible (0), positions 4-7 are masked (-inf)

row 4: [0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf, -inf]
       positions 0-4 are visible (0), positions 5-7 are masked (-inf)

row 5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf, -inf]
       positions 0-5 are visible (0), positions 6-7 are masked (-inf)

row 6: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inf]
       positions 0-6 are visible (0), position 7 is masked (-inf)

row 7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
       all positions 0-7 are visible (0), nothing is masked

```


```python
# apply softmax
# softmax converts values to probabilities that sum to 1
# e^(-inf) = 0, so masked positions become 0 probability
# dim=-1 means apply softmax along the last dimension (each row)
wei = F.softmax(wei, dim=-1)
wei
```


**Output:**
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```


```python
# understand why e^(-inf) = 0
print('understanding why the -inf trick works')
print()
print('softmax formula: softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)')
print()
print('key insight: e^(-inf) = 0')
print()
import math
print('examples')
print(f'   e^(0) = {math.exp(0):.4f}')
print(f'   e^(1) = {math.exp(1):.4f}')
print(f'   e^(-1) = {math.exp(-1):.4f}')
print(f'   e^(-10) = {math.exp(-10):.10f}')
print(f'   e^(-100) = {math.exp(-100):.50f}')
print('   e^(-inf) = 0 (exactly)')
print()
print('so when we set future positions to -inf')
print('they become e^(-inf) = 0 in the softmax numerator')
print('this means they contribute 0 probability!')
```


**Output:**
```
understanding why the -inf trick works

softmax formula: softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)

key insight: e^(-inf) = 0

examples
   e^(0) = 1.0000
   e^(1) = 2.7183
   e^(-1) = 0.3679
   e^(-10) = 0.0000453999
   e^(-100) = 0.00000000000000000000000000000000000000000003720076
   e^(-inf) = 0 (exactly)

so when we set future positions to -inf
they become e^(-inf) = 0 in the softmax numerator
this means they contribute 0 probability!

```


```python
# trace through softmax for each row manually
print('tracing softmax for each row')
print()
print('row 0: input = [0, -inf, -inf, -inf, -inf, -inf, -inf, -inf]')
print('   e^0 = 1, e^(-inf) = 0, e^(-inf) = 0, ...')
print('   numerators = [1, 0, 0, 0, 0, 0, 0, 0]')
print('   sum = 1')
print('   softmax = [1/1, 0/1, 0/1, ...] = [1.0, 0, 0, 0, 0, 0, 0, 0]')
print(f'   actual: {wei[0].tolist()}')
print()
print('row 1: input = [0, 0, -inf, -inf, -inf, -inf, -inf, -inf]')
print('   e^0 = 1, e^0 = 1, e^(-inf) = 0, ...')
print('   numerators = [1, 1, 0, 0, 0, 0, 0, 0]')
print('   sum = 2')
print('   softmax = [1/2, 1/2, 0/2, ...] = [0.5, 0.5, 0, 0, 0, 0, 0, 0]')
print(f'   actual: {wei[1].tolist()}')
print()
print('row 2: input = [0, 0, 0, -inf, -inf, -inf, -inf, -inf]')
print('   e^0 = 1, e^0 = 1, e^0 = 1, e^(-inf) = 0, ...')
print('   numerators = [1, 1, 1, 0, 0, 0, 0, 0]')
print('   sum = 3')
print('   softmax = [1/3, 1/3, 1/3, 0, ...] ≈ [0.333, 0.333, 0.333, 0, 0, 0, 0, 0]')
print(f'   actual: {wei[2].tolist()}')
```


**Output:**
```
tracing softmax for each row

row 0: input = [0, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
   e^0 = 1, e^(-inf) = 0, e^(-inf) = 0, ...
   numerators = [1, 0, 0, 0, 0, 0, 0, 0]
   sum = 1
   softmax = [1/1, 0/1, 0/1, ...] = [1.0, 0, 0, 0, 0, 0, 0, 0]
   actual: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

row 1: input = [0, 0, -inf, -inf, -inf, -inf, -inf, -inf]
   e^0 = 1, e^0 = 1, e^(-inf) = 0, ...
   numerators = [1, 1, 0, 0, 0, 0, 0, 0]
   sum = 2
   softmax = [1/2, 1/2, 0/2, ...] = [0.5, 0.5, 0, 0, 0, 0, 0, 0]
   actual: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

row 2: input = [0, 0, 0, -inf, -inf, -inf, -inf, -inf]
   e^0 = 1, e^0 = 1, e^0 = 1, e^(-inf) = 0, ...
   numerators = [1, 1, 1, 0, 0, 0, 0, 0]
   sum = 3
   softmax = [1/3, 1/3, 1/3, 0, ...] ≈ [0.333, 0.333, 0.333, 0, 0, 0, 0, 0]
   actual: [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.0, 0.0, 0.0, 0.0, 0.0]

```


```python
# continue tracing softmax for remaining rows
print('continuing softmax trace')
print()
print('row 3: input = [0, 0, 0, 0, -inf, -inf, -inf, -inf]')
print('   numerators = [1, 1, 1, 1, 0, 0, 0, 0]')
print('   sum = 4')
print('   softmax = [1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0] = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]')
print(f'   actual: {wei[3].tolist()}')
print()
print('row 4: input = [0, 0, 0, 0, 0, -inf, -inf, -inf]')
print('   numerators = [1, 1, 1, 1, 1, 0, 0, 0]')
print('   sum = 5')
print('   softmax = [1/5, 1/5, 1/5, 1/5, 1/5, 0, 0, 0] = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]')
print(f'   actual: {wei[4].tolist()}')
print()
print('row 5: input = [0, 0, 0, 0, 0, 0, -inf, -inf]')
print('   numerators = [1, 1, 1, 1, 1, 1, 0, 0]')
print('   sum = 6')
print('   softmax = [1/6, ...] ≈ [0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0, 0]')
print(f'   actual: {wei[5].tolist()}')
print()
print('row 6: input = [0, 0, 0, 0, 0, 0, 0, -inf]')
print('   numerators = [1, 1, 1, 1, 1, 1, 1, 0]')
print('   sum = 7')
print('   softmax = [1/7, ...] ≈ [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0]')
print(f'   actual: {wei[6].tolist()}')
print()
print('row 7: input = [0, 0, 0, 0, 0, 0, 0, 0]')
print('   numerators = [1, 1, 1, 1, 1, 1, 1, 1]')
print('   sum = 8')
print('   softmax = [1/8, ...] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]')
print(f'   actual: {wei[7].tolist()}')
```


**Output:**
```
continuing softmax trace

row 3: input = [0, 0, 0, 0, -inf, -inf, -inf, -inf]
   numerators = [1, 1, 1, 1, 0, 0, 0, 0]
   sum = 4
   softmax = [1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0] = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]
   actual: [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]

row 4: input = [0, 0, 0, 0, 0, -inf, -inf, -inf]
   numerators = [1, 1, 1, 1, 1, 0, 0, 0]
   sum = 5
   softmax = [1/5, 1/5, 1/5, 1/5, 1/5, 0, 0, 0] = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]
   actual: [0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.0, 0.0, 0.0]

row 5: input = [0, 0, 0, 0, 0, 0, -inf, -inf]
   numerators = [1, 1, 1, 1, 1, 1, 0, 0]
   sum = 6
   softmax = [1/6, ...] ≈ [0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0, 0]
   actual: [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.0, 0.0]

row 6: input = [0, 0, 0, 0, 0, 0, 0, -inf]
   numerators = [1, 1, 1, 1, 1, 1, 1, 0]
   sum = 7
   softmax = [1/7, ...] ≈ [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0]
   actual: [0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.0]

row 7: input = [0, 0, 0, 0, 0, 0, 0, 0]
   numerators = [1, 1, 1, 1, 1, 1, 1, 1]
   sum = 8
   softmax = [1/8, ...] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
   actual: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

```


```python
# verify each row sums to 1
print('verify: each row sums to 1.0')
print()
for i in range(T):
    row_sum = wei[i].sum().item()
    print(f'row {i} sum: {row_sum:.4f}')
```


**Output:**
```
verify: each row sums to 1.0

row 0 sum: 1.0000
row 1 sum: 1.0000
row 2 sum: 1.0000
row 3 sum: 1.0000
row 4 sum: 1.0000
row 5 sum: 1.0000
row 6 sum: 1.0000
row 7 sum: 1.0000

```


```python
# why softmax with masking is used in real attention
print('💡 Why softmax with masking?')
print('   - In real attention, "wei" will not be all zeros')
print('   - It will have LEARNED, data-dependent values')
print('   - But we still need to mask future → -inf trick works!')
print()
print('example: imagine learned attention weights')
print('   raw weights = [0.5, 0.8, 0.2, 0.9, ...]')
print('   after masking row 1 = [0.5, 0.8, -inf, -inf, ...]')
print('   softmax only considers 0.5 and 0.8')
print('   result = [e^0.5 / (e^0.5 + e^0.8), e^0.8 / (e^0.5 + e^0.8), 0, 0, ...]')
print()
print('the -inf masking trick works regardless of what the original values are')
```


**Output:**
```
💡 Why softmax with masking?
   - In real attention, "wei" will not be all zeros
   - It will have LEARNED, data-dependent values
   - But we still need to mask future → -inf trick works!

example: imagine learned attention weights
   raw weights = [0.5, 0.8, 0.2, 0.9, ...]
   after masking row 1 = [0.5, 0.8, -inf, -inf, ...]
   softmax only considers 0.5 and 0.8
   result = [e^0.5 / (e^0.5 + e^0.8), e^0.8 / (e^0.5 + e^0.8), 0, 0, ...]

the -inf masking trick works regardless of what the original values are

```


```python
# apply to get weighted averages
# wei @ x performs the weighted averaging
# wei shape: (T, T) = (8, 8)
# x shape: (B, T, C) = (4, 8, 2)
# result shape: (B, T, C) = (4, 8, 2)
x_bow_3 = wei @ x
x_bow_3
```


**Output:**
```
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])
```


### Understanding the Matrix Multiplication with Softmax Weights
When we do `wei @ x`, PyTorch broadcasts the operation.
- `wei` has shape (T, T) = (8, 8)
- `x` has shape (B, T, C) = (4, 8, 2)

PyTorch treats the batch dimension (B=4) specially. It performs 4 separate matrix multiplications.
- `wei @ x[0]` → result for batch 0
- `wei @ x[1]` → result for batch 1  
- `wei @ x[2]` → result for batch 2
- `wei @ x[3]` → result for batch 3

For each batch, the multiplication is the following.
- (8, 8) @ (8, 2) = (8, 2)

The final result has shape (4, 8, 2) = (B, T, C).


```python
# let's trace through the matrix multiplication step by step for batch 0
print('understanding the matrix multiplication for batch 0')
print()
print(f'wei shape: {wei.shape}')
print(f'x[0] shape: {x[0].shape}')
print()
print('x[0] (the input for batch 0)')
print(x[0])
print()
print('wei (the softmax weight matrix)')
print(wei)
```


**Output:**
```
understanding the matrix multiplication for batch 0

wei shape: torch.Size([8, 8])
x[0] shape: torch.Size([8, 2])

x[0] (the input for batch 0)
tensor([[ 1.9269,  1.4873],
        [ 0.9007, -2.1055],
        [ 0.6784, -1.2345],
        [-0.0431, -1.6047],
        [-0.7521,  1.6487],
        [-0.3925, -1.4036],
        [-0.7279, -0.5594],
        [-0.7688,  0.7624]])

wei (the softmax weight matrix)
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

```


```python
# position 0 calculation (row 0 of wei @ x[0])
print('position 0 calculation')
print()
print(f'wei[0] = {wei[0].tolist()}')
print(f'this means: 1.0 * x[0,0] + 0.0 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]')
print()
print('for feature 0')
val = wei[0, 0].item() * x[0, 0, 0].item()
print(f'   {wei[0, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val:.4f}')
print()
print('for feature 1')
val = wei[0, 0].item() * x[0, 0, 1].item()
print(f'   {wei[0, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val:.4f}')
print()
print(f'result: x_bow_3[0, 0] = {x_bow_3[0, 0].tolist()}')
print(f'verify: x[0, 0]       = {x[0, 0].tolist()}')
print('(position 0 just equals itself since it only sees itself)')
```


**Output:**
```
position 0 calculation

wei[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 1.0 * x[0,0] + 0.0 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]

for feature 0
   1.0000 * 1.9269 = 1.9269

for feature 1
   1.0000 * 1.4873 = 1.4873

result: x_bow_3[0, 0] = [1.9269150495529175, 1.4872841835021973]
verify: x[0, 0]       = [1.9269150495529175, 1.4872841835021973]
(position 0 just equals itself since it only sees itself)

```


```python
# position 1 calculation (row 1 of wei @ x[0])
print('position 1 calculation')
print()
print(f'wei[1] = {wei[1].tolist()}')
print(f'this means: 0.5 * x[0,0] + 0.5 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]')
print()
print('for feature 0')
val0 = wei[1, 0].item() * x[0, 0, 0].item()
val1 = wei[1, 1].item() * x[0, 1, 0].item()
print(f'   {wei[1, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val0:.4f}')
print(f' + {wei[1, 1].item():.4f} * {x[0, 1, 0].item():.4f} = {val1:.4f}')
print(f'   sum = {val0 + val1:.4f}')
print()
print('for feature 1')
val0 = wei[1, 0].item() * x[0, 0, 1].item()
val1 = wei[1, 1].item() * x[0, 1, 1].item()
print(f'   {wei[1, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val0:.4f}')
print(f' + {wei[1, 1].item():.4f} * {x[0, 1, 1].item():.4f} = {val1:.4f}')
print(f'   sum = {val0 + val1:.4f}')
print()
print(f'result: x_bow_3[0, 1] = {x_bow_3[0, 1].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1]) / 2
print(f'(x[0,0] + x[0,1]) / 2 = {manual_avg.tolist()}')
```


**Output:**
```
position 1 calculation

wei[1] = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 0.5 * x[0,0] + 0.5 * x[0,1] + 0.0 * x[0,2] + ... + 0.0 * x[0,7]

for feature 0
   0.5000 * 1.9269 = 0.9635
 + 0.5000 * 0.9007 = 0.4504
   sum = 1.4138

for feature 1
   0.5000 * 1.4873 = 0.7436
 + 0.5000 * -2.1055 = -1.0528
   sum = -0.3091

result: x_bow_3[0, 1] = [1.4138160943984985, -0.3091186285018921]

manual verification
(x[0,0] + x[0,1]) / 2 = [1.4138160943984985, -0.3091186285018921]

```


```python
# position 2 calculation (row 2 of wei @ x[0])
print('position 2 calculation')
print()
print(f'wei[2] = {wei[2].tolist()}')
print(f'this means: 0.333 * x[0,0] + 0.333 * x[0,1] + 0.333 * x[0,2] + 0.0 * x[0,3] + ...')
print()
print('for feature 0')
val0 = wei[2, 0].item() * x[0, 0, 0].item()
val1 = wei[2, 1].item() * x[0, 1, 0].item()
val2 = wei[2, 2].item() * x[0, 2, 0].item()
print(f'   {wei[2, 0].item():.4f} * {x[0, 0, 0].item():.4f} = {val0:.4f}')
print(f' + {wei[2, 1].item():.4f} * {x[0, 1, 0].item():.4f} = {val1:.4f}')
print(f' + {wei[2, 2].item():.4f} * {x[0, 2, 0].item():.4f} = {val2:.4f}')
print(f'   sum = {val0 + val1 + val2:.4f}')
print()
print('for feature 1')
val0 = wei[2, 0].item() * x[0, 0, 1].item()
val1 = wei[2, 1].item() * x[0, 1, 1].item()
val2 = wei[2, 2].item() * x[0, 2, 1].item()
print(f'   {wei[2, 0].item():.4f} * {x[0, 0, 1].item():.4f} = {val0:.4f}')
print(f' + {wei[2, 1].item():.4f} * {x[0, 1, 1].item():.4f} = {val1:.4f}')
print(f' + {wei[2, 2].item():.4f} * {x[0, 2, 1].item():.4f} = {val2:.4f}')
print(f'   sum = {val0 + val1 + val2:.4f}')
print()
print(f'result: x_bow_3[0, 2] = {x_bow_3[0, 2].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2]) / 3
print(f'(x[0,0] + x[0,1] + x[0,2]) / 3 = {manual_avg.tolist()}')
```


**Output:**
```
position 2 calculation

wei[2] = [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.0, 0.0, 0.0, 0.0, 0.0]
this means: 0.333 * x[0,0] + 0.333 * x[0,1] + 0.333 * x[0,2] + 0.0 * x[0,3] + ...

for feature 0
   0.3333 * 1.9269 = 0.6423
 + 0.3333 * 0.9007 = 0.3002
 + 0.3333 * 0.6784 = 0.2261
   sum = 1.1687

for feature 1
   0.3333 * 1.4873 = 0.4958
 + 0.3333 * -2.1055 = -0.7018
 + 0.3333 * -1.2345 = -0.4115
   sum = -0.6176

result: x_bow_3[0, 2] = [1.168683648109436, -0.6175941228866577]

manual verification
(x[0,0] + x[0,1] + x[0,2]) / 3 = [1.1686835289001465, -0.6175940632820129]

```


```python
# position 3 calculation (row 3 of wei @ x[0])
print('position 3 calculation')
print()
print(f'wei[3] = {wei[3].tolist()}')
print(f'this means: 0.25 * x[0,0] + 0.25 * x[0,1] + 0.25 * x[0,2] + 0.25 * x[0,3] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[3, i].item() * x[0, i, 0].item() for i in range(4)]
for i in range(4):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[3, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[3, i].item() * x[0, i, 1].item() for i in range(4)]
for i in range(4):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[3, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_3[0, 3] = {x_bow_3[0, 3].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3]) / 4
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3]) / 4 = {manual_avg.tolist()}')
```


**Output:**
```
position 3 calculation

wei[3] = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
this means: 0.25 * x[0,0] + 0.25 * x[0,1] + 0.25 * x[0,2] + 0.25 * x[0,3] + 0.0 * ...

for feature 0
   0.2500 * 1.9269 = 0.4817
 + 0.2500 * 0.9007 = 0.2252
 + 0.2500 * 0.6784 = 0.1696
 + 0.2500 * -0.0431 = -0.0108
   sum = 0.8657

for feature 1
   0.2500 * 1.4873 = 0.3718
 + 0.2500 * -2.1055 = -0.5264
 + 0.2500 * -1.2345 = -0.3086
 + 0.2500 * -1.6047 = -0.4012
   sum = -0.8644

result: x_bow_3[0, 3] = [0.8657457828521729, -0.8643622994422913]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3]) / 4 = [0.8657457828521729, -0.8643622994422913]

```


```python
# position 4 calculation (row 4 of wei @ x[0])
print('position 4 calculation')
print()
print(f'wei[4] = {wei[4].tolist()}')
print(f'this means: 0.2 * x[0,0] + 0.2 * x[0,1] + 0.2 * x[0,2] + 0.2 * x[0,3] + 0.2 * x[0,4] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[4, i].item() * x[0, i, 0].item() for i in range(5)]
for i in range(5):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[4, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[4, i].item() * x[0, i, 1].item() for i in range(5)]
for i in range(5):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[4, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_3[0, 4] = {x_bow_3[0, 4].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4]) / 5
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4]) / 5 = {manual_avg.tolist()}')
```


**Output:**
```
position 4 calculation

wei[4] = [0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.0, 0.0, 0.0]
this means: 0.2 * x[0,0] + 0.2 * x[0,1] + 0.2 * x[0,2] + 0.2 * x[0,3] + 0.2 * x[0,4] + 0.0 * ...

for feature 0
   0.2000 * 1.9269 = 0.3854
 + 0.2000 * 0.9007 = 0.1801
 + 0.2000 * 0.6784 = 0.1357
 + 0.2000 * -0.0431 = -0.0086
 + 0.2000 * -0.7521 = -0.1504
   sum = 0.5422

for feature 1
   0.2000 * 1.4873 = 0.2975
 + 0.2000 * -2.1055 = -0.4211
 + 0.2000 * -1.2345 = -0.2469
 + 0.2000 * -1.6047 = -0.3209
 + 0.2000 * 1.6487 = 0.3297
   sum = -0.3617

result: x_bow_3[0, 4] = [0.542169451713562, -0.36174529790878296]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4]) / 5 = [0.5421693921089172, -0.36174526810646057]

```


```python
# position 5 calculation (row 5 of wei @ x[0])
print('position 5 calculation')
print()
print(f'wei[5] = {wei[5].tolist()}')
print(f'this means: 0.167 * x[0,0] + 0.167 * x[0,1] + ... + 0.167 * x[0,5] + 0.0 * ...')
print()
print('for feature 0')
vals = [wei[5, i].item() * x[0, i, 0].item() for i in range(6)]
for i in range(6):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[5, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[5, i].item() * x[0, i, 1].item() for i in range(6)]
for i in range(6):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[5, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_3[0, 5] = {x_bow_3[0, 5].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5]) / 6
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5]) / 6 = {manual_avg.tolist()}')
```


**Output:**
```
position 5 calculation

wei[5] = [0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.1666666716337204, 0.0, 0.0]
this means: 0.167 * x[0,0] + 0.167 * x[0,1] + ... + 0.167 * x[0,5] + 0.0 * ...

for feature 0
   0.1667 * 1.9269 = 0.3212
 + 0.1667 * 0.9007 = 0.1501
 + 0.1667 * 0.6784 = 0.1131
 + 0.1667 * -0.0431 = -0.0072
 + 0.1667 * -0.7521 = -0.1254
 + 0.1667 * -0.3925 = -0.0654
   sum = 0.3864

for feature 1
   0.1667 * 1.4873 = 0.2479
 + 0.1667 * -2.1055 = -0.3509
 + 0.1667 * -1.2345 = -0.2058
 + 0.1667 * -1.6047 = -0.2674
 + 0.1667 * 1.6487 = 0.2748
 + 0.1667 * -1.4036 = -0.2339
   sum = -0.5354

result: x_bow_3[0, 5] = [0.38639479875564575, -0.5353888869285583]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5]) / 6 = [0.3863947093486786, -0.5353888869285583]

```


```python
# position 6 calculation (row 6 of wei @ x[0])
print('position 6 calculation')
print()
print(f'wei[6] = {wei[6].tolist()}')
print(f'this means: 0.143 * x[0,0] + 0.143 * x[0,1] + ... + 0.143 * x[0,6] + 0.0 * x[0,7]')
print()
print('for feature 0')
vals = [wei[6, i].item() * x[0, i, 0].item() for i in range(7)]
for i in range(7):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[6, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[6, i].item() * x[0, i, 1].item() for i in range(7)]
for i in range(7):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[6, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_3[0, 6] = {x_bow_3[0, 6].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5] + x[0, 6]) / 7
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6]) / 7 = {manual_avg.tolist()}')
```


**Output:**
```
position 6 calculation

wei[6] = [0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.1428571492433548, 0.0]
this means: 0.143 * x[0,0] + 0.143 * x[0,1] + ... + 0.143 * x[0,6] + 0.0 * x[0,7]

for feature 0
   0.1429 * 1.9269 = 0.2753
 + 0.1429 * 0.9007 = 0.1287
 + 0.1429 * 0.6784 = 0.0969
 + 0.1429 * -0.0431 = -0.0062
 + 0.1429 * -0.7521 = -0.1074
 + 0.1429 * -0.3925 = -0.0561
 + 0.1429 * -0.7279 = -0.1040
   sum = 0.2272

for feature 1
   0.1429 * 1.4873 = 0.2125
 + 0.1429 * -2.1055 = -0.3008
 + 0.1429 * -1.2345 = -0.1764
 + 0.1429 * -1.6047 = -0.2292
 + 0.1429 * 1.6487 = 0.2355
 + 0.1429 * -1.4036 = -0.2005
 + 0.1429 * -0.5594 = -0.0799
   sum = -0.5388

result: x_bow_3[0, 6] = [0.22721239924430847, -0.5388233065605164]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6]) / 7 = [0.22721242904663086, -0.5388233065605164]

```


```python
# position 7 calculation (row 7 of wei @ x[0])
print('position 7 calculation')
print()
print(f'wei[7] = {wei[7].tolist()}')
print(f'this means: 0.125 * x[0,0] + 0.125 * x[0,1] + ... + 0.125 * x[0,7]')
print()
print('for feature 0')
vals = [wei[7, i].item() * x[0, i, 0].item() for i in range(8)]
for i in range(8):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[7, i].item():.4f} * {x[0, i, 0].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print('for feature 1')
vals = [wei[7, i].item() * x[0, i, 1].item() for i in range(8)]
for i in range(8):
    prefix = '   ' if i == 0 else ' + '
    print(f'{prefix}{wei[7, i].item():.4f} * {x[0, i, 1].item():.4f} = {vals[i]:.4f}')
print(f'   sum = {sum(vals):.4f}')
print()
print(f'result: x_bow_3[0, 7] = {x_bow_3[0, 7].tolist()}')
print()
print('manual verification')
manual_avg = (x[0, 0] + x[0, 1] + x[0, 2] + x[0, 3] + x[0, 4] + x[0, 5] + x[0, 6] + x[0, 7]) / 8
print(f'(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7]) / 8 = {manual_avg.tolist()}')
```


**Output:**
```
position 7 calculation

wei[7] = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
this means: 0.125 * x[0,0] + 0.125 * x[0,1] + ... + 0.125 * x[0,7]

for feature 0
   0.1250 * 1.9269 = 0.2409
 + 0.1250 * 0.9007 = 0.1126
 + 0.1250 * 0.6784 = 0.0848
 + 0.1250 * -0.0431 = -0.0054
 + 0.1250 * -0.7521 = -0.0940
 + 0.1250 * -0.3925 = -0.0491
 + 0.1250 * -0.7279 = -0.0910
 + 0.1250 * -0.7688 = -0.0961
   sum = 0.1027

for feature 1
   0.1250 * 1.4873 = 0.1859
 + 0.1250 * -2.1055 = -0.2632
 + 0.1250 * -1.2345 = -0.1543
 + 0.1250 * -1.6047 = -0.2006
 + 0.1250 * 1.6487 = 0.2061
 + 0.1250 * -1.4036 = -0.1755
 + 0.1250 * -0.5594 = -0.0699
 + 0.1250 * 0.7624 = 0.0953
   sum = -0.3762

result: x_bow_3[0, 7] = [0.10270600765943527, -0.3761647045612335]

manual verification
(x[0,0] + x[0,1] + x[0,2] + x[0,3] + x[0,4] + x[0,5] + x[0,6] + x[0,7]) / 8 = [0.10270600765943527, -0.3761647045612335]

```


```python
# print shapes summary
print('version 3: softmax with masking averaging')
print()
print(f'wei shape:    {wei.shape} → (T={T}, T={T})')
print(f'x shape:      {x.shape} → (B={B}, T={T}, C={C})')
print(f'result shape: {x_bow_3.shape} → (B={B}, T={T}, C={C})')
print()
print('Same output shape as input!')
print('Each position now holds the average of itself and all previous positions.')
```


**Output:**
```
version 3: softmax with masking averaging

wei shape:    torch.Size([8, 8]) → (T=8, T=8)
x shape:      torch.Size([4, 8, 2]) → (B=4, T=8, C=2)
result shape: torch.Size([4, 8, 2]) → (B=4, T=8, C=2)

Same output shape as input!
Each position now holds the average of itself and all previous positions.

```


### Comparing All Three Versions
All three methods produce the EXACT same result! Let's verify this.


```python
# recreate version 1 result using for-loops (from Part 3)
print('recreating version 1 (for-loop method) for comparison')
print()
x_bow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        x_previous = x[b, :t+1]
        x_bow[b, t] = torch.mean(x_previous, dim=0)
print('x_bow (for-loop result)')
print(x_bow)
```


**Output:**
```
recreating version 1 (for-loop method) for comparison

x_bow (for-loop result)
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])

```


```python
# recreate version 2 result using matrix multiplication (from Part 4)
print('recreating version 2 (matrix multiplication method) for comparison')
print()
wei_v2 = torch.tril(torch.ones(T, T))
wei_v2 = wei_v2 / wei_v2.sum(dim=1, keepdim=True)
x_bow_2 = wei_v2 @ x
print('x_bow_2 (matrix multiplication result)')
print(x_bow_2)
```


**Output:**
```
recreating version 2 (matrix multiplication method) for comparison

x_bow_2 (matrix multiplication result)
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])

```


```python
# compare all three versions
print('comparing all three versions')
print()
print('x_bow_3 (softmax with masking result)')
print(x_bow_3)
print()
print('Are version 1 and version 3 equal?')
print(f'torch.allclose(x_bow, x_bow_3) = {torch.allclose(x_bow, x_bow_3)}')
print()
print('Are version 2 and version 3 equal?')
print(f'torch.allclose(x_bow_2, x_bow_3) = {torch.allclose(x_bow_2, x_bow_3)}')
print()
print('exact difference (should be all zeros or very close)')
diff = x_bow - x_bow_3
print(f'max absolute difference (v1 vs v3): {torch.abs(diff).max().item()}')
diff2 = x_bow_2 - x_bow_3
print(f'max absolute difference (v2 vs v3): {torch.abs(diff2).max().item()}')
```


**Output:**
```
comparing all three versions

x_bow_3 (softmax with masking result)
tensor([[[ 1.9269,  1.4873],
         [ 1.4138, -0.3091],
         [ 1.1687, -0.6176],
         [ 0.8657, -0.8644],
         [ 0.5422, -0.3617],
         [ 0.3864, -0.5354],
         [ 0.2272, -0.5388],
         [ 0.1027, -0.3762]],

        [[ 1.6423, -0.1596],
         [ 0.5725,  0.1400],
         [ 0.1289,  0.4528],
         [ 0.2969,  0.7597],
         [ 0.4933,  0.8671],
         [ 0.5129,  0.9450],
         [ 0.4065,  0.8160],
         [ 0.3242,  0.8215]],

        [[-1.3847, -0.8712],
         [-0.8040,  0.4231],
         [-0.4297,  0.1405],
         [-0.2459, -0.0882],
         [-0.5082,  0.1285],
         [-0.5701,  0.0069],
         [-0.6707,  0.3092],
         [-0.7412,  0.2095]],

        [[-0.9138, -0.6581],
         [-0.4179, -0.0662],
         [-0.4413,  0.3530],
         [-0.5344,  0.0808],
         [-0.7082,  0.0718],
         [-0.6008,  0.1724],
         [-0.5289,  0.4113],
         [-0.6109,  0.5329]]])

Are version 1 and version 3 equal?
torch.allclose(x_bow, x_bow_3) = True

Are version 2 and version 3 equal?
torch.allclose(x_bow_2, x_bow_3) = True

exact difference (should be all zeros or very close)
max absolute difference (v1 vs v3): 1.1920928955078125e-07
max absolute difference (v2 vs v3): 0.0

```


```python
# element by element comparison for batch 0
print('element by element comparison for batch 0')
print()
for t in range(T):
    print(f'position {t}')
    print(f'   for-loop result:  {x_bow[0, t].tolist()}')
    print(f'   matrix result:    {x_bow_2[0, t].tolist()}')
    print(f'   softmax result:   {x_bow_3[0, t].tolist()}')
    print(f'   all match: {torch.allclose(x_bow[0, t], x_bow_3[0, t]) and torch.allclose(x_bow_2[0, t], x_bow_3[0, t])}')
    print()
```


**Output:**
```
element by element comparison for batch 0

position 0
   for-loop result:  [1.9269150495529175, 1.4872841835021973]
   matrix result:    [1.9269150495529175, 1.4872841835021973]
   softmax result:   [1.9269150495529175, 1.4872841835021973]
   all match: True

position 1
   for-loop result:  [1.4138160943984985, -0.3091186285018921]
   matrix result:    [1.4138160943984985, -0.3091186285018921]
   softmax result:   [1.4138160943984985, -0.3091186285018921]
   all match: True

position 2
   for-loop result:  [1.1686835289001465, -0.6175940632820129]
   matrix result:    [1.168683648109436, -0.6175941228866577]
   softmax result:   [1.168683648109436, -0.6175941228866577]
   all match: True

position 3
   for-loop result:  [0.8657457828521729, -0.8643622994422913]
   matrix result:    [0.8657457828521729, -0.8643622994422913]
   softmax result:   [0.8657457828521729, -0.8643622994422913]
   all match: True

position 4
   for-loop result:  [0.542169451713562, -0.36174526810646057]
   matrix result:    [0.542169451713562, -0.36174529790878296]
   softmax result:   [0.542169451713562, -0.36174529790878296]
   all match: True

position 5
   for-loop result:  [0.386394739151001, -0.5353888869285583]
   matrix result:    [0.38639479875564575, -0.5353888869285583]
   softmax result:   [0.38639479875564575, -0.5353888869285583]
   all match: True

position 6
   for-loop result:  [0.22721245884895325, -0.5388233065605164]
   matrix result:    [0.22721239924430847, -0.5388233065605164]
   softmax result:   [0.22721239924430847, -0.5388233065605164]
   all match: True

position 7
   for-loop result:  [0.10270603746175766, -0.37616467475891113]
   matrix result:    [0.10270600765943527, -0.3761647045612335]
   softmax result:   [0.10270600765943527, -0.3761647045612335]
   all match: True


```


### Why Softmax with Masking is Used in Real Transformers
| Aspect | Division (Version 2) | Softmax + Masking (Version 3) |
|--------|---------------------|------------------------------|
| Fixed Weights | Yes (1/n for all) | No (can be any distribution) |
| Learnable | No | Yes (input values can be learned) |
| Differentiable | Yes | Yes |
| Future Masking | Via tril structure | Via -inf masking |
| Real Attention | No | Yes (this is the pattern used) |

Softmax with masking is the foundation of causal self-attention because it allows LEARNED, data-dependent weights while still preventing future token leakage.


```python
# final summary: the complete softmax + masking approach
print('SUMMARY: Softmax with Masking for Token Averaging')
print('=' * 60)
print()
print('step 1: create lower triangular mask')
print('        tril = torch.tril(torch.ones(T, T))')
print('        this identifies which positions can be seen')
print()
print('step 2: start with initial weights (zeros or learned values)')
print('        wei = torch.zeros((T, T))')
print('        in real attention, these come from query-key dot products')
print()
print('step 3: mask future positions with -inf')
print('        wei = wei.masked_fill(tril == 0, float("-inf"))')
print('        e^(-inf) = 0, so future positions get 0 probability')
print()
print('step 4: apply softmax')
print('        wei = F.softmax(wei, dim=-1)')
print('        converts to probabilities that sum to 1')
print()
print('step 5: matrix multiply')
print('        x_bow_3 = wei @ x')
print('        applies the weighted average')
print()
print('Result: Same as all previous versions!')
print()
print('This is the foundation of causal self-attention.')
print('In real transformers, step 2 uses learned query-key products')
print('instead of zeros, allowing dynamic, content-based attention.')
```


**Output:**
```
SUMMARY: Softmax with Masking for Token Averaging
============================================================

step 1: create lower triangular mask
        tril = torch.tril(torch.ones(T, T))
        this identifies which positions can be seen

step 2: start with initial weights (zeros or learned values)
        wei = torch.zeros((T, T))
        in real attention, these come from query-key dot products

step 3: mask future positions with -inf
        wei = wei.masked_fill(tril == 0, float("-inf"))
        e^(-inf) = 0, so future positions get 0 probability

step 4: apply softmax
        wei = F.softmax(wei, dim=-1)
        converts to probabilities that sum to 1

step 5: matrix multiply
        x_bow_3 = wei @ x
        applies the weighted average

Result: Same as all previous versions!

This is the foundation of causal self-attention.
In real transformers, step 2 uses learned query-key products
instead of zeros, allowing dynamic, content-based attention.

```


## MIT License

