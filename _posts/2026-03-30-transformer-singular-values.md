---
title: "Singular Value Structure of Transformer Attention Heads"
date: 2026-03-30
excerpt: "An empirical study of the singular value spectra of W_QK matrices across transformer architectures: spectral distributions, participation ratios, and what they reveal about learned attention geometry."
series: "transformer-spin"
series_order: 3
permalink: /posts/2026/transformer-singular-values/
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

The [previous post](/posts/2026/transformer-analysis/) established the basic distributional facts about transformer weight matrices: the element-wise statistics of $W_Q$, $W_K$, and $W_{QK}$ across several architectures, how the trained distributions depart from their Gaussian initialization, and the amplification of non-Gaussianity in the matrix product $W_{QK} = W_Q^T W_K$. Those results characterized the marginal, element-wise statistics of these matrices. This post moves to the spectral structure: the singular value decomposition of $W_{QK}$, which captures the geometry of the learned attention patterns rather than just the distribution of individual matrix elements.

<!-- TODO: add a sentence reminding readers about the interactive dashboard for exploring these results -->

Recall that $W_{QK}$ is a $d \times d$ matrix of rank at most $d_h = d/n_h$, where $d_h$ is the head dimension. Its singular values $\{\lambda_k\}$ define the effective interaction strengths in the attention logit computation. In a physics context, these are the mode energies of the coupling matrix. The distribution of these modes, whether concentrated in a few dominant directions or spread across many, encodes how many effective degrees of freedom each attention head uses and how the learned attention geometry differs from the random-matrix baseline.


The random-matrix expectations were developed in the previous post: for $W_{QK}$ constructed from Gaussian $W_Q$ and $W_K$, singular value spectrum is very similar to a gapped Marchenko-Pastur (MP) distribution with $d_h$ nonzero values. Note that the MP distribution is usually expressed as the distribution on the eigenvalues of .... with the eignvalues equal to the square of the singular values. 

Any departure from this baseline in trained models is, by construction, a signal of learned structure. To make this comparison more rigorous, Monte Carlo simulations of random matrices were utilized to generate the appropriate matrix ensemble. These results are described in more detail and cataloged in the a future post. For now we use these simulations to sharpen the quantitative expectations for our singular value study.

The simulated distribution of eigenvalues and spectrum are shown below using $d$=768 and $d_h$=64, just as in gpt2. These results assume an underlying Gaussian $\sigma$ of 1. While the spectrum shares many features of the Marchenko-Pastur distribution, the empirical distribution is wider owing to ... text from appendix... For some statistics the product expansion moment method can be used to perform analytical calculations for this model and those are also shown and agree well with the simulated values.

The next figure shows the same quantities but now plotted in terms of of the singular values. 



## Singular Value Distributions

### Single-Head Spectra

![Figure 1: MC eigenvalue distribution](/images/transformer-analysis/sv_mc_baseline.png)

<p class="fig-caption">Figure 1: Eigenvalue ($\lambda^2$) distribution from Monte Carlo simulation of the random-matrix baseline (GPT-2 dimensions). Left: full distribution with Marchenko–Pastur density overlay and bounds. Right: zoom on the upper edge showing MC support beyond the MP limit, well-captured by the exact (moment-method) calculation.</p>

<!-- TODO: prose on what this figure establishes — the MC baseline, how it departs from naive MP, role of exact bounds -->

To ground these expectations against trained models, we compare two specific attention heads that illustrate the range of spectral behavior. The first has a singular value spectrum that closely follows the random-matrix baseline; the second exhibits clear outlier singular values emerging from the bulk.

In the left figure, the blue exhibits a clear excess at low index: the largest singular values in this head are larger than in the other, which directly contributes to the points in the right figure outside the bulk. The kinks in the distribution near index = 4 and $d_h$ correspond to the bulk with the slope of the distribution determining the curvature of the bulk in the $P(\lambda)$, the peak at zero corresponds to the zero singular values, which are present due to the low rank structure: there should be $d_h = 64$ non-zero singular values.

The red distribution exhibits a tighter bulk distribution, with a few marginally significant deviations at the upper edge of the range. Both distributions retain enough of the original randomness that a gap in the spectrum is still present. We will see that for some heads the kink points effectively merge, destroying the gap and giving more of a power law fall off. 


![Figure 2: Combined single-head comparison](/images/transformer-analysis/sv_single_head_comparison.png)

<p class="fig-caption">Figure 2: Singular value spectra for two GPT-2 Large attention heads. Left: singular values vs. index. Right: spectral density $P(\lambda)$, log scale. The MP-like head (L7 H6, blue) closely follows the random-matrix baseline, while the outlier head (L1 H0, red) shows dominant low-rank structure.</p>

<!-- TODO: prose on what "MP-like" means physically; interpretation of outlier SVs -->

### Spectra Across the Full Model

To summarize the spectral structure across the full model, the figures below show the $W_{QK}$ singular values for every attention head as 2D heatmaps with layer and head on the vertical axis.

The largest singular values are generally concentrated in the earliest layers reaching values $\lambda_{\max} \lesssim 50$. These heads also generally show a steeper slope, attaining smaller values of $\lambda_{\min}$.

This can be more easily seen in the next figure which shows the spectra density versus $\lambda$. Generally most distributions retain a gap, with some distributions collapsing more to the power law spectrum alluded to earlier. The distributions are both broader with more outliers in the earlier layers, although there are some intermediate layers that show similar behavior.

![Figure 3: SV index heatmap](/images/transformer-analysis/sv_heatmap_svd.png)

<p class="fig-caption">Figure 3: Singular values vs. index across all attention heads in GPT-2, shown as a heatmap on log color scale. Each row is one head, grouped by layer.</p>

![Figure 4: P(lambda) heatmap](/images/transformer-analysis/sv_heatmap_plambda.png)

<p class="fig-caption">Figure 4: Spectral density $P(\lambda)$ across all attention heads in GPT-2, log color scale. Color scale is set to exclude the zero-bin peak.</p>

<!-- TODO: describe what the heatmaps reveal — layer trends, head-to-head variation, any outlier heads -->

### Per-Head Distributions

The per-head grid provides a closer look at the spectral diversity within a single layer.

![Figure 5: Per-head SVD at layer 0](/images/transformer-analysis/sv_layer_grid_svd.png)

<p class="fig-caption">Figure 5: Singular values vs. index for all 12 attention heads in GPT-2 layer 0.</p>

![Figure 6: Per-head spectral density at layer 0](/images/transformer-analysis/sv_layer_grid_plambda.png)

<p class="fig-caption">Figure 6: Spectral density $P(\lambda)$ for all 12 attention heads in GPT-2 layer 0. Different heads develop fundamentally different spectral shapes, suggesting distinct functional roles.</p>

<!-- TODO: interpret — do different heads show qualitatively different shapes? flat vs peaked? any evidence of low effective rank? -->





## Singular Value Statistics

The full singular value spectrum contains $d_h$ values per head. To track how spectral structure varies across layers and models, we compress these into scalar summary statistics. 

### Leading Singular Value

The largest singular value $\lambda_0$ sets the overall scale of the attention logits for each head. Its variation across layers reveals the model's allocation of attention strength. The layer/head dependence of this statistic follows from the previous observations about the change in the distribution. The layer average generally descreases until about the $d_h / 2 = 18$, after which point the trend is marginal, although there are hints that the distribution may rise in the last few layers. A trend that is present in other statistics, including the standard deviation, and across models.

![Figure 7: Leading singular value across layers](/images/transformer-analysis/sv_leading_lambda_vs_layer.png)

<p class="fig-caption">Figure 7: Leading singular value $\lambda_0$ of $W_{QK}$ across all attention heads in GPT-2, ordered by layer. Dashed red line shows layer average. <!-- TODO: describe trend --></p>

### Participation Ratio and Normalized Participation Ratio

The participation ratio $\text{PR} = (\sum_k \lambda_k)^2 / \sum_k \lambda_k^2$ measures how many singular values contribute meaningfully to the spectrum. For a flat spectrum (all $\lambda_k$ equal), $\text{PR} = d_h$; for a single dominant mode, $\text{PR} = 1$. The normalized participation ratio $\text{NPR} = \text{PR}/d_h$ maps this to $[0, 1]$. Since the MC prediction for this quantity is independent of the underlying $\sigma$, requiring fewer assumptions, it is included here. 

At large NPR all the modes are contributing equally, lots of nearly degenerate eigenvalues, one big mode described by multiple small ones, these merge into larger modes 

![Figure 8: Normalized participation ratio across layers](/images/transformer-analysis/sv_npr_vs_layer.png)

<p class="fig-caption">Figure 8: Normalized participation ratio of $W_{QK}$ singular values across all attention heads in GPT-2. Dashed red line shows layer average. <!-- TODO: describe layer dependence; NPR near 1 for GPT-2 but gets smaller for larger models --></p>

### Spectral Entropy

The spectral entropy $S_\lambda = -\sum_k p_k \ln p_k$, where $p_k = \lambda_k^2 / \sum_j \lambda_j^2$, provides a related but distinct measure of spectral concentration. While the participation ratio counts effective modes, the entropy is more sensitive to the shape of the distribution across those modes.

![Figure 9: Spectral entropy across layers](/images/transformer-analysis/sv_spectral_entropy_vs_layer.png)

<p class="fig-caption">Figure 9: Spectral entropy $S_\lambda$ of $W_{QK}$ across all attention heads in GPT-2. Dashed red line shows layer average. <!-- TODO: describe --></p>

### Condition Number

The condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ (computed over the $d_h$ nonzero singular values) measures the dynamic range of the spectrum. A condition number near 1 indicates a flat, democratic spectrum; large values indicate that some modes are orders of magnitude stronger than others. For GPT-2 (small), typical condition numbers are around 4, but other models exhibit dramatically larger values that warrant investigation.

![Figure 10: Condition number across layers](/images/transformer-analysis/sv_condition_number_vs_layer.png)

<p class="fig-caption">Figure 10: Condition number $\kappa$ of $W_{QK}$ across all attention heads in GPT-2 (log scale). Dashed red line shows layer average. <!-- TODO: describe; note GPT-2 small ~ 4, some other models blow up --></p>

### Entropy vs. Participation Ratio

The relationship between spectral entropy and the normalized participation ratio provides a consistency check: both measure spectral concentration, but weight the tails differently.

![Figure 11: Spectral entropy vs. normalized participation ratio](/images/transformer-analysis/sv_entropy_vs_npr.png)

<p class="fig-caption">Figure 11: Spectral entropy vs. normalized participation ratio for all GPT-2 attention heads, colored by layer. <!-- TODO: describe correlation structure --></p>


## Correlations with Element-Wise Statistics

The singular value statistics developed above characterize the spectral geometry of $W_{QK}$; the element-wise statistics from the previous post characterize its marginal distribution. These are not independent descriptions. The standard deviation $\sigma$ sets the overall scale of the matrix elements and therefore controls the scale of the singular values. The question is how much of the spectral structure is determined by $\sigma$ alone, and how much is independent.

![Figure 12: SV statistics vs. element-wise sigma](/images/transformer-analysis/sv_stats_vs_sigma.png)

<p class="fig-caption">Figure 12: Singular value statistics vs. element-wise $\sigma$ for all attention heads across models. Top left: $\lambda_0$ (messy linear with width). Top right: $\Sigma\lambda$ (linear). Bottom left: $\Sigma\lambda^2$ (quadratic, cleanest relationship). Bottom right: spectral entropy.</p>

<!-- TODO: check whether entropy or KL divergence from post 1 adds predictive power beyond sigma -->

## Cross-Model Comparison

![Figure 13: Cross-model leading SV](/images/transformer-analysis/sv_cross_model_leading_lambda.png)

<p class="fig-caption">Figure 13: Leading singular value $\lambda_0$ (layer-averaged) across layers for several model families. <!-- TODO: describe --></p>

![Figure 14: Cross-model NPR comparison](/images/transformer-analysis/sv_cross_model_npr.png)

<p class="fig-caption">Figure 14: Normalized participation ratio (layer-averaged) across layers for several model families. <!-- TODO: describe trends — NPR near 1 for GPT-2 but gets smaller for larger models --></p>

![Figure 15: Cross-model spectral entropy](/images/transformer-analysis/sv_cross_model_spectral_entropy.png)

<p class="fig-caption">Figure 15: Spectral entropy $S_\lambda$ (layer-averaged) across layers for several model families. <!-- TODO: describe --></p>

![Figure 16: Cross-model condition number](/images/transformer-analysis/sv_cross_model_condition_number.png)

<p class="fig-caption">Figure 16: Condition number $\kappa$ (layer-averaged, log scale) across layers for several model families. <!-- TODO: describe; note GPT-2 small ~ 4, GPT-2 medium blows up to ~24M (dead eigenvalues?), LLaMA and some Pythias also large --></p>

<!-- TODO: does the Mistral-LLaMA sigma anomaly from post 1 manifest in the SV statistics as well? -->

## Discussion

<!-- TODO: synthesize the main findings:
  - What does NPR near 1 mean physically? (heads use nearly all available dimensions)
  - What does lower NPR in larger models mean? (more specialization, lower effective rank)
  - How do the SV statistics connect back to the element-wise non-Gaussianity from post 1?
  - Any heads that are clear outliers in both spectral and element-wise measures?
-->

## What Comes Next

<!-- TODO: preview post 3 — training dynamics from Pythia checkpoints, how SV structure emerges during training -->

The code, data, and interactive dashboards for reproducing and extending this analysis are available at the links above.
