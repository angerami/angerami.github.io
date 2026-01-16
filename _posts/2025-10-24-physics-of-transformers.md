---
title: "The Physics of Transformers"
date: 2025-10-24
excerpt: "Recasting the equations of this core ML architectural element in terms of spin systems."
series: "transformer-spin"      # Must match across all posts in series
series_order: 1
permalink: /posts/2025/physics-of-transformers/
tags:
  - transformers
  - statistical-physics
published: false
draft: true
---

{% include wip-notice.html %}
{% include transformer-spin-preamble.html %}

## Introduction

## Transformer Basics
### Sequence Transduction and Terminology

Transformers perform a sequence transduction task, taking a sequence of inputs and producing a sequence of outputs. Applied to natural language processing, this means that an input sequence, such as a sentence or paragraph, is parsed into a sequence of words or tokens.

The tokens are then identified with elements of a dictionary, resulting in a quantity $t_i^a$, where
- The index $i$: $i=1,\,\cdots \Nseq$ identifies which element of the sequence we are looking at
- The index $a$: $a=1,\,\cdots \Nvocab$ identifies which element of the dictionary the token corresponds to, for exmample you could assign $t_i^a = a$, assigning the value to be the index of the word in the vocabulary.

These are very long vectors, but they have _discrete_ values. A more compact representation is to use an embedding, where the elements are real-valued vectors of length $\dmodel < \Nvocab$, which is the dimension of the model.

Following this embedding, the sequence is transformed via an operation called _positional encoding_, which adds to each element $i$ some information about the other elements in the sequence ($j\neq i$).

Both of these operations can be motivated and understood from a physics perspective, but for now we will defer that discussion and focus on middle chunk of the transformer. Thus in what follows, $x_i^a$ will refer to our sequence after embedding and positional encoding.

### The Transformer Equations
First we consider a single transformer block, for the moment ignoring the possibility of having multiple attention heads. The often-quoted equation from Attention is All You Need is:

| Name | Compact Notation | Index Notation |
|:-----|:----------------:|:--------------:|
| Query | $\QQ \equiv \WQ \vv{x}$ | $Q_i^{a} \equiv \sum_{b} \left[\WQ\right]^{ab} x_i^{b}$ |
| Key | $\KK \equiv \WK \vv{x}$ | $K_i^{a} \equiv \sum_{b} \left[\WK\right]^{ab} x_i^{b}$ |
| Value | $\VV \equiv \WV \vv{x}$ | $V_i^{a} \equiv \sum_{b} \left[\WV\right]^{ab} x_i^{b}$ |
| Attention Score | $\omega = \KK^{T} \cdot \QQ$ | $\omega_{ij} \equiv \sum_{a,\,b} x_j^{a} \left[{\WK}^{T} {\WQ}\right]^{ab}  x_i^{b}$ |
| Attention Weight | $\alpha = \softmax(\omega)$ | $\alpha_{ij} \equiv \exp{\left(\omega_{ij}/\sqrt{\dmodel}\right)} / \sum_{k} \exp{\left(\omega_{ik}/\sqrt{\dmodel}\right)}$ |
| Attention Output | $\vv{z} \equiv \alpha \VV$ | $z_{i}^{a} = \sum_{j} \alpha_{ij} \sum_{b} [\WV]^{ab}x_{j}^{b}$|

The $\mm{W}$ are matrices in the model space, at present these are $\dmodel \times \dmodel$ matrices, but we will explore variations later. The second column shows standard quantity definitions in NLP literature (vector multiplication over model dimension, sequence index suppressed), while the right column shows all indices explicitly. I've adopted the the convention of writing sequence index as a subscript, indexed using $i,\,j,\,k\,,\cdots$, and the feature vector indices as superscripts labelled $a,\,b,\,c\,,\cdots$. I'm going to call these ``spin'' indices.

I like this form since it reminds me that the normalization for the softmax involves a sum over $j$, and you can easily see how the attention weights are not symmetric in $i$ and $j$. 

In a transformer model the $\mm{W}$ matrices are learnable parameters.

### Transformers Transformed

In the expression for the attention score $\omega$, we see that the matrix product over $\WQ$ and $\WK$ can be performed without reference to the input vectors $\vv{x}$. As we shall see later, the introduction of separate matrices for $\WQ$ and $\WK$ plays an important role, however for the present purposes, it obfuscates the underlying physics interpretation. 

$$
 \left[{\WK}^{T} {\WQ}\right]^{ab} = \sum_{c} \left[{\WK}^{T}\right]^{ac} \left[{\WQ}\right]^{cb} \equiv [\WQK]^{ab}
 $$
 
For now we note that this term looks just like the expression for the Hamiltonian for a system of interacting spins.

$$
H^{\text{spin}}_{ij} = {\vv{x}}^{T}_{i} \mm{J} {\vv{x}}^{T}_{j} 
$$

where $\mm{J} \rightarrow \WQK / \sqrt{\dmodel}$.

## Conclusion and Outlook