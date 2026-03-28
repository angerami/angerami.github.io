---
title: "Weight Distributions in Transformer Attention Heads"
date: 2026-01-16
excerpt: "An empirical study of the statistical distributions of W_Q, W_K, and W_QK weight matrices across transformer architectures."
series: "transformer-spin"
series_order: 2
permalink: /posts/2026/transformer-analysis/
tags:
  - transformers
  - statistical-physics
published: true
draft: true
---
<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-bottom: 20px;">
<strong>⚠️ Work in Progress</strong> - This post is still being developed and may contain incomplete sections or change substantially.
</div>

[![Code](https://img.shields.io/badge/📁-Code_on_Github-blue)](https://github.com/angerami/transformer-analysis)
[![Dashboard](https://img.shields.io/badge/📊-Interactive_Dashboard-orange)](https://huggingface.co/spaces/angerami/transformer-weights)
[![Data](https://img.shields.io/badge/🗂️-Datasets_on_HuggingFace-yellow)](https://huggingface.co/angerami)

## Introduction

I've been interested in applying a physics framework to transformers involving dynamical systems and statistical mechanics. In this approach, the feature vectors representing tokens are interpreted as spins, the model weights play the role of interaction strengths between different token vectors, and the self-attention mechanism maps onto a thermodynamic average over these interactions. The structure of the model, with multiple attention heads, layers, and residual connections, allows for a large number of distinct, metastable equilibrium configurations: a spin glass system.

The object at the center of this picture is the bilinear form governing the attention logits for a single head,

$$
\vec{Q}_{i}^{T} \cdot \vec{K_j} \,\,= \,\,\vec{x_i}^{T}W_{Q}^{T}W_{K}\vec{x_{j}} \,\,\equiv\,\, \vec{x_i}^{T}\, W_{QK}\, \vec{x_{j}}
$$

where $W_{QK} = W_{Q}^{T}W_{K}$ is the matrix that directly determines pairwise token interactions. In a physical system, this would be the coupling matrix specifying the energy of interaction between two spins. Understanding its statistical structure means understanding the geometry of the learned attention patterns.

At initialization, each attention head has the elements of $W_{Q}$ and $W_{K}$ drawn from some initial distribution, usually $\mathcal{N}(0,\sigma_0)$. The distribution for $W_{QK}$ is not set directly but rather follows from the distributions of the underlying weight matrices. Over the course of training the distribution changes shape to encode the information learned during back-propagation. This post examines the empirical distributions of these weight matrices across several transformer architectures and asks: how do the learned weight distributions differ from their initial Gaussian form, and what does this reveal about what models learn?

This is the first in a series of posts exploring transformer weights from a statistical physics perspective. Here I focus on establishing the basic distributional facts: the element-wise statistics of $W_Q$, $W_K$, and $W_{QK}$ for trained models, how these vary across layers and heads, and what distinguishes different model architectures. Subsequent posts will examine the singular value structure of $W_{QK}$ and the training dynamics revealed by the Pythia checkpoint suite.

## The Low-Rank Structure of $W_{QK}$

In the typical transformer architecture, $W_{Q}$ and $W_{K}$ are $d_h \times d$ matrices, with $d_h = d / n_h$. Their product $W_{QK}$ is therefore a $d \times d$ matrix, but it has rank at most $d_h$. This low-rank constraint means that the smallest $d - d_h$ singular values are identically zero, and $W_{QK}$ operates only within a $d_h$-dimensional subspace of the full residual stream.

The effective rank of the matrix depends on the alignment of the singular vectors of $W_{Q}$ and $W_{K}$. These structures evolve dynamically during training and the actual rank of $W_{QK}$ for a trained model is emergent rather than fixed by the architecture. In a physics context, we try to associate the singular values of $W_{QK}$ with modes of definite energy, corresponding to particular directions in the spin space. The distribution of these modes encodes the effective degrees of freedom available to each attention head.

To develop some baseline intuition, I begin with a few numerical examples. The figures below show three-panel summaries of random matrices at $d$ = 1000: the element-wise probability density $P(W)$, the singular values ordered by index, and the spectral density $P(\lambda)$. All curves are averaged over multiple random draws.

The first figure shows full-rank random matrices with elements drawn from $\mathcal{N}(0,\sigma)$ for several values of $\sigma$. The element-wise distributions are, tautologically, Gaussian, and the singular values follow the Marchenko-Pastur distribution from random matrix theory.

![Figure 1: Full-rank random matrices](/images/transformer-analysis/random_matrix_full_rank.png)

<p class="fig-caption">Figure 1: Full-rank random matrices at $d$ = 1000 for several values of $\sigma$. Left: element-wise density $P(W)$. Center: singular values vs. index. Right: spectral density $P(\lambda)$.</p>

The next figure introduces the low-rank structure relevant to $W_{QK}$. Setting $W_{QK} = W_{Q}^{T}W_{K}$ with $d_h = d/2$ = 500, the element-wise distribution narrows and the singular value spectrum develops a sharp transition at the $d_h$-th index, dropping effectively to zero. The spectral density develops a gapped structure with support over two disconnected domains.

![Figure 2: Low-rank random matrix products](/images/transformer-analysis/random_matrix_low_rank.png)

<p class="fig-caption">Figure 2: Low-rank random matrix products $W_{QK} = W_Q^T W_K$ at $d$ = 1000 and $d_h$ = 500, for several values of $\sigma$. The singular value spectrum exhibits a sharp cutoff at $d_h$, and the spectral density develops a gap.</p>

Varying $d_h$ while keeping $\sigma$ = 1 reveals the expected behavior: as the rank decreases (more heads), the gap widens and the element-wise distribution narrows. The element-wise standard deviation scales as $\sigma_{W_{QK}} \propto \sqrt{d_h / d} = 1/\sqrt{n_h}$, and the entropy follows $S = \frac{1}{2}\ln(2\pi e / n_h)$. The spectral edge positions are given approximately by the Marchenko-Pastur bounds: $\lambda_{\pm} = \sqrt{d_h}(1 \pm \sqrt{d / d_h})$.

![Figure 3: Effect of varying rank](/images/transformer-analysis/random_matrix_varying_rank.png)

<p class="fig-caption">Figure 3: Effect of varying $d_h$ at fixed $d$ = 1000 and $\sigma$ = 1. Left: element-wise distributions narrow as $1/\sqrt{n_h}$. Center: singular values with cutoff at $d_h$. Right: spectral density transitions from gapped (low rank) to continuous (full rank).</p>

These baselines establish what we should expect from random, untrained weight matrices. Deviations from these predictions in trained models are the signal of learned structure.

## Models and Data

The analysis uses pre-trained models available on HuggingFace. For each model, I extract the $W_Q$, $W_K$, and $W_{QK}$ matrices for every attention head across all layers, then compute element-wise histograms, summary statistics (mean, standard deviation, skewness, kurtosis), entropy measures, and the full singular value decomposition.

| Model | $d$ | $n_h$ | Layers | Parameters | Notes |
|-------|:---:|:---:|:---:|:---:|-------|
| GPT-2 (small) | 768 | 12 | 12 | 124M | Primary illustration model |
| GPT-2-XL | 1600 | 25 | 48 | 1.5B | Larger variant for comparison |
| Pythia-70M | 512 | 8 | 6 | 70M | Smallest Pythia model |
| Pythia-1.4B | 2048 | 16 | 24 | 1.4B | Mid-range Pythia |
| Pythia-2.8B | 2560 | 32 | 32 | 2.8B | Large Pythia |
| LLaMA 3 8B | 4096 | 32 | 32 | 8B | Uses grouped-query attention |
| Mistral 7B v3 | 4096 | 32 | 32 | 7B | Uses grouped-query attention |

<p class="fig-caption">Table 1: Models included in this study. All models are publicly available on HuggingFace. LLaMA and Mistral use grouped-query attention (GQA), where the number of K/V heads is smaller than Q heads; K heads are expanded via <code>repeat_interleave</code> to match the Q head count before computing $W_{QK}$.</p>

The Pythia model suite, developed by EleutherAI, provides an additional dimension: 154 training checkpoints from initialization to convergence, enabling the study of how these distributions evolve during training. The time-evolution analysis is deferred to a subsequent post, but the Pythia models appear in the cross-model comparisons below.

All datasets are published on HuggingFace under `angerami/weight_study_ana-003` (cross-model) and `angerami/pythia-{size}-deduped_weight_evolution_001` (training dynamics). An interactive Streamlit dashboard for exploring the data is available on [HuggingFace Spaces](https://huggingface.co/spaces/angerami/transformer-weights).

## Results: Weight Distributions in GPT-2

### Per-Head Distributions

To establish some expectations we start with GPT-2 (small), which has $d$ = 768, 12 layers, and 12 attention heads per layer. The $W_Q$ and $W_K$ matrices each contain 768 $\times$ 768 / 12 $\approx$ 49,000 elements per head, providing a substantial statistical sample.

The figure below shows histograms of the $W_Q$ weight distribution for the 12 attention heads in layer 0. Each of these can be thought of as a separate system of $\sim$ 49,000 degrees of freedom. At initialization all heads are drawn from the same distribution $\mathcal{N}(0, \sigma_0)$, but as you can see they evolve differently during training.

![Figure 4: W_Q weight distributions for 12 attention heads in GPT-2 layer 0](/images/transformer-analysis/gpt2_wq_layer0_linear.png)

<p class="fig-caption">Figure 4: $W_Q$ weight distributions for 12 attention heads in GPT-2 layer 0 (linear scale). Despite identical initialization, the trained distributions develop distinct widths and shapes.</p>

The same distributions on a logarithmic vertical scale, with Gaussian fits overlaid, reveal not only that the distributions have different $\sigma$ values but that they show different levels of deviation from normality. Some heads remain well-described by a Gaussian; others develop heavier tails, particularly on the negative side.

![Figure 5: Same distributions on logarithmic scale with Gaussian fits](/images/transformer-analysis/gpt2_wq_layer0_log.png)

<p class="fig-caption">Figure 5: Same distributions as Figure 4, on logarithmic scale with Gaussian fits. Deviations from the Gaussian envelope, visible as excess probability in the tails, vary substantially across heads.</p>

### Distributions Across the Full Model

To summarize the entire model, the figure below shows the $W_Q$ distributions for all heads across all layers as a 2D heatmap, where each row corresponds to one attention head (grouped by layer) and the color indicates log probability density. The distributions generally narrow as one goes deeper in the network. While different heads in the same layer tend to be similar, the figure reveals a few cases where particular heads show very different behavior from these general trends.

![Figure 6: W_Q heatmap across all heads and layers of GPT-2](/images/transformer-analysis/gpt2_wq_heatmap.png)

<p class="fig-caption">Figure 6: $W_Q$ weight distributions across all heads and layers of GPT-2 (small), shown as a 2D heatmap on logarithmic color scale. Each row is one head; layers are separated by dashed lines. The general trend is narrowing with depth, but individual heads can deviate substantially.</p>

### Summary Statistics

Two summary statistics, the standard deviation $\sigma$ and the differential entropy $S = -\int P(w) \ln P(w)\, dw$ (estimated via KDE), confirm the behavior observed above. The distributions qualitatively narrow with depth, but there are large variations among heads within the same layer that are of a similar scale as the overall downward trend.

![Figure 7: Summary statistics across GPT-2 attention heads](/images/transformer-analysis/gpt2_wq_summary_stats.png)

<p class="fig-caption">Figure 7: Standard deviation (blue) and entropy (red) across all attention heads in GPT-2 (small), ordered by layer. Vertical dashed lines separate layers. Both statistics show a general decrease with depth, modulated by substantial head-to-head variation within each layer.</p>

### GPT-2 XL

The same analysis for GPT-2 XL ($d$ = 1600, 25 heads, 48 layers) provides a useful comparison. The 2$\times$ increase in model dimension results in a 4$\times$ increase in the number of matrix elements per head, which should reduce sampling fluctuations on statistical estimators by roughly 2$\times$. The summary statistics plot does indeed look smoother. However, while this model exhibits some of the same trends as GPT-2 (small), there is a notable resurgence in the later layers: the standard deviation and entropy, after decreasing through the middle layers, increase again toward the output.

![Figure 8: Summary statistics for GPT-2 XL](/images/transformer-analysis/gpt2xl_wq_summary_stats.png)

<p class="fig-caption">Figure 8: Summary statistics for GPT-2 XL. Unlike GPT-2 (small), the standard deviation and entropy show a non-monotonic pattern with a resurgence in later layers.</p>

## Results: $W_{QK}$ Distributions

### Amplification of Non-Gaussianity

The element-wise distributions of $W_Q$ and $W_K$ individually remain approximately Gaussian across most heads and layers. Their product $W_{QK}$, however, tells a different story. Small deviations from normality in the individual matrices are dramatically amplified by the matrix product. The $W_{QK}$ distributions develop heavy tails, particularly on the negative side, that grow much faster than those in the individual matrices.

![Figure 9: W_QK distributions for GPT-2 layer 0](/images/transformer-analysis/gpt2_wqk_layer0_log.png)

<p class="fig-caption">Figure 9: $W_{QK}$ weight distributions for GPT-2 layer 0 on logarithmic scale. Compared to the individual $W_Q$ distributions (Figure 5), the deviations from Gaussianity are substantially larger, with pronounced asymmetric tails.</p>

This asymmetry is physically interesting. The $W_{QK}$ matrix is the pre-softmax attention logit matrix, so its statistical structure directly encodes what the model has learned about which tokens should attend to which. A heavier negative tail means more extreme "don't attend" signals than "do attend" signals. In the language of the softmax function, strong negative logits are effectively suppressed to zero attention weight, so the model appears to learn structured inhibition patterns.

### $W_{QK}$ Across the Full Model

The 2D heatmap for $W_{QK}$ shows more dramatic structure than the individual weight matrices. The variation across layers and heads is more pronounced, and the tails (visible as extended color in the heatmap wings) are clearly wider than what a Gaussian with the same $\sigma$ would produce.

![Figure 10: W_QK heatmap across GPT-2](/images/transformer-analysis/gpt2_wqk_heatmap.png)

<p class="fig-caption">Figure 10: $W_{QK}$ distributions across all heads and layers of GPT-2 (small). The non-Gaussian character is visible as extended tails (color at large $|w|$) beyond what the central peak width would predict.</p>

## Cross-Model Comparison

Comparing across model families reveals both universal features and notable differences. The general pattern, where $W_Q$ and $W_K$ remain approximately Gaussian while $W_{QK}$ develops heavier tails, is consistent across all architectures studied. However, the scale and shape of the distributions vary substantially.

### The Mistral-LLaMA Sigma Anomaly

Despite identical architecture parameters ($d$ = 4096, 32 heads, 32 layers), Mistral's $W_Q$ and $W_K$ standard deviations are 5--10$\times$ smaller than LLaMA's, and this difference compounds in $W_{QK}$. Both models use the same `initializer_range` of 0.02.

The leading hypothesis is that the two models use different normalization factors in the attention scaling. If one model uses $1/\sqrt{d}$ while the other uses $1/\sqrt{d_h}$, the ratio of effective scales is $\sqrt{d/d_h} = \sqrt{4096/128} \approx$ 5.66, which matches the observed ratio nearly exactly. This finding illustrates that models with identical parameter counts and architecture can exhibit fundamentally different weight characteristics after training, suggesting different optimization landscapes.

![Figure 11: Cross-model sigma comparison](/images/transformer-analysis/cross_model_sigma.png)

<p class="fig-caption">Figure 11: Standard deviation of $W_Q$ across layers for several model families. The Mistral-LLaMA scale difference is clearly visible despite identical architectural dimensions.</p>

## What Comes Next

This post establishes the basic distributional facts: individual weight matrices $W_Q$ and $W_K$ remain well-behaved and approximately Gaussian, but their product $W_{QK}$ develops structured non-Gaussianity with heavy, asymmetric tails. These deviations from the random-matrix baselines are the signal of learned structure.

The next post in this series examines the singular value decomposition of $W_{QK}$, which moves beyond the element-wise marginal statistics explored here to characterize the spectral structure: how many effective dimensions each head uses, how the spectrum differs from the Marchenko-Pastur predictions, and what this reveals about the geometry of learned attention patterns. The third post will use the Pythia training checkpoints to study how all of these structures emerge during training.

The code, data, and interactive dashboards for reproducing and extending this analysis are available at the links above.
