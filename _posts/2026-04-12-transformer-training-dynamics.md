---
title: "Training Dynamics of Transformer Attention Heads"
date: 2026-04-12
excerpt: "A time-dependent study of W_QK statistics across training checkpoints in the Pythia model suite: how spectral structure, participation ratios, and head diversity evolve during pretraining."
series: "transformer-spin"
series_order: 3
permalink: /posts/2026/transformer-training-dynamics/
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

The [previous posts](/posts/2026/transformer-singular-values/) in this series characterized the weight matrix statistics and spectral structure of $W_{QK}$ in trained models: how the element-wise distributions depart from Gaussian initialization, how the singular value spectra of attention heads vary across layers, and how summary statistics like the normalized participation ratio and condition number scale with model size. Those analyses were snapshots of final, trained weights. This post adds the temporal dimension.

The Pythia model suite provides an unusual resource for this: a set of models spanning 70M to 12B parameters, each with approximately 154 publicly released checkpoints taken at regular intervals across pretraining on the Pile. Rather than inferring training history from the endpoint, these checkpoints allow the spectral statistics of $W_{QK}$ to be tracked directly as a function of training step. The reference model throughout is Pythia-410M ($d = 1024$, $n_h = 16$, $d_h = 64$, 24 layers), which is large enough to exhibit diverse head behavior while remaining computationally tractable for a full checkpoint sweep.

The central questions are: when during training does the spectral structure of $W_{QK}$ consolidate? Do different heads converge on their final geometry at different rates? And does the layer-dependent organization identified in the static analysis (e.g. early layers developing the most concentrated spectra),emerge early in training or gradually across the full pretraining horizon?

The interactive dashboard linked above supports direct exploration of these statistics across the Pythia suite.

## Basic Results



### Element-Wise Distributions Across Training

<!-- TODO: decide whether to lead with W_Q or jump straight to W_QK. Post-3 notes say "undecided". For now, stub shows W_QK directly. -->

The element-wise statistics of $W_{QK}$ evolve substantially during the early phase of training and then stabilize. The figure below shows the two-dimensional distribution of matrix elements at a fixed head and layer at three representative checkpoints: early training (near initialization), mid-training, and the final checkpoint.

![Figure 1: 2D element-wise distributions at three training stages](/images/transformer-analysis/tp_2d_distribution.png)

<p class="fig-caption">Figure 1: Element-wise distribution of $W_{QK}$ for a representative head (Pythia-410M, layer 0, head 0) at three training checkpoints. [PLACEHOLDER — to be updated with actual figure.]</p>

The transition from the near-Gaussian initialization to the structured, heavier-tailed trained distribution is rapid: by a few thousand steps the qualitative shape has shifted, though the amplitude continues to grow. The one-dimensional slices through this joint distribution make the sharpening of structure more legible.

![Figure 2: 1D slices at fixed head across training steps](/images/transformer-analysis/tp_1d_slices.png)

<p class="fig-caption">Figure 2: 1D marginal distributions of $W_{QK}$ elements at selected training steps. The distribution narrows from its initialized width and then reorganizes as training proceeds. [PLACEHOLDER]</p>

### Statistics vs. Training Step

A more quantitative picture of this evolution comes from tracking scalar summary statistics. The figure below shows the standard deviation $\sigma$ of $W_{QK}$ elements as a function of training step for a fixed head and layer. This is representative of the behavior of other statistics and illustrates the qualitative types of trajectories observed across the model.

![Figure 3: Standard deviation vs. training step, fixed head](/images/transformer-analysis/tp_stat_vs_step.png)

<p class="fig-caption">Figure 3: Element-wise standard deviation $\sigma$ of $W_{QK}$ vs. training step for a fixed head and layer (Pythia-410M). Dashed line shows the final checkpoint value. [PLACEHOLDER]</p>

The trajectory shows a rapid initial rise followed by a plateau, with the timescale of convergence varying across heads and layers. Early-layer heads tend to reach their asymptotic values faster and at larger absolute $\sigma$; later-layer heads show more extended evolution. Whether this reflects the early layers acting as a more stable attractor or simply a larger gradient signal from the output is not resolved by these statistics alone.

### Time Evolution Across All Heads

The heatmap below extends this view to all heads in the model simultaneously, with training step on the horizontal axis and head index (sorted by layer) on the vertical axis.

![Figure 4: Heatmap of sigma across all heads and training steps](/images/transformer-analysis/tp_heatmap_all_heads.png)

<p class="fig-caption">Figure 4: Standard deviation $\sigma$ of $W_{QK}$ across all 384 attention heads in Pythia-410M (16 heads × 24 layers), shown as a function of training step. Color scale is log. [PLACEHOLDER]</p>

Several features are apparent. The early-layer heads (top rows) develop larger $\sigma$ values earlier. The within-layer spread — visible as column-wise variation within each layer block — increases over training and does not disappear: the heads within a layer diversify rather than converging toward a common value. This is consistent with the functional specialization hypothesis, in which heads within a layer develop distinct roles that are reflected in their weight statistics.

### Head Diversity Within a Layer

The multi-panel view below highlights this within-layer diversity. Each panel shows the time trajectory of $\sigma$ for a single head in one layer, making it easy to compare how quickly different heads within the same layer differentiate.

![Figure 5: Multi-head comparison within a single layer](/images/transformer-analysis/tp_multi_head_panel.png)

<p class="fig-caption">Figure 5: Time trajectories of $\sigma$ for all 16 heads in a single layer of Pythia-410M. Heads within the same layer develop qualitatively different training dynamics. [PLACEHOLDER]</p>

Some heads rise monotonically to their final value; others exhibit non-monotone behavior with an early overshoot that partially relaxes. A few heads remain near their initialized values for a substantial fraction of training before rapidly converging. This heterogeneity within a single layer suggests that the dynamics of individual attention heads are not well-described by a single timescale or trajectory type.

### UMAP of Training Trajectories

To visualize the diversity of head trajectories more globally, we apply UMAP to the time series of $\sigma$ across all heads, embedding each head's training trajectory as a point in two dimensions. Heads whose $\sigma$ evolves in similar ways appear nearby in this embedding.

![Figure 6: UMAP of training trajectories across all heads](/images/transformer-analysis/tp_umap_timecurves.png)

<p class="fig-caption">Figure 6: UMAP embedding of training-step trajectories of $\sigma$ for all attention heads in Pythia-410M. Points are colored by layer. Distinct trajectory types are visible as clusters. [PLACEHOLDER — see src/post_analysis/step-behavior-study.py for sketch]</p>

The embedding reveals a small number of qualitatively distinct trajectory types, with layer as a strong predictor of cluster membership but not a perfect one. Heads from the same layer can belong to different clusters, confirming that layer alone does not determine training dynamics. Whether these clusters correspond to functional head types — induction heads, positional heads, and so on — is a natural question for future work.

## Time-Dependent Spectral Analysis

The spectral statistics from the [singular value post](/posts/2026/transformer-singular-values/) can be tracked as a function of training step in exactly the same way as the element-wise statistics above. The three-panel figure below summarizes the spectral evolution at a fixed head, showing the element-wise distribution $P(w)$, the singular value distribution, and the eigenvalue density $P(\lambda)$ at early, middle, and late training checkpoints.

![Figure 7: Spectral evolution at a fixed head](/images/transformer-analysis/tp_sv_threepanel.png)

<p class="fig-caption">Figure 7: Spectral structure of $W_{QK}$ at three training stages for a fixed head (Pythia-410M). Left: element-wise distribution $P(w)$. Center: singular value spectrum. Right: eigenvalue density $P(\lambda)$ with Marchenko–Pastur overlay. [PLACEHOLDER]</p>

Early in training, the singular value spectrum closely follows the random-matrix baseline: the eigenvalue distribution is consistent with Marchenko-Pastur at the appropriate $\sigma$. As training proceeds, the leading eigenvalues grow above the bulk while the smaller eigenvalues compress, widening the spectral gap. By the final checkpoint, the spectrum has taken on the structured form described in the previous post — a clear gap, a dominant outlier region, and a bulk whose shape depends on the specific head.

The normalized participation ratio and spectral entropy follow the same qualitative trajectory: both start near their random-matrix predictions and decrease as training reinforces a smaller set of dominant directions. The timescale of this specialization varies across heads in the same way as the element-wise statistics, and the final NPR is correlated with the layer and with the trajectory type identified in the UMAP embedding.

## What Comes Next

<!-- TODO: discuss whether post 4 will cover additional Pythia model sizes for scale-dependent dynamics, or shift to the W_Q/W_K decomposition. See post-4.md. -->

The analyses here use Pythia-410M as a reference, but the checkpoint data covers the full Pythia suite. A natural extension is to ask whether the trajectory types identified here — the timescales, the UMAP cluster structure, the pattern of early-layer consolidation — are universal across model scales or change systematically with parameter count. The Pythia suite, with its consistent training setup across scales, provides a controlled setting for that comparison.

The code, data, and interactive dashboards for reproducing and extending this analysis are available at the links above.
