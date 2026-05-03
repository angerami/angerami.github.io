---
layout: archive
title: "Transformer-Spin Correspondence"
permalink: /posts/transformer-spin/
author_profile: true
---

{% include base_path %}

## About This Series

This project treats transformer weights as a dynamical system. The work develops a statistical analysis pipeline that ingests trained models and extracts a set of interpretable metrics to characterize their behavior and track its evolution during training. The framework supports both an empirical characterization of the weights and a physics-motivated interpretation grounded in the correspondence between transformer self-attention and the statistical mechanics of spin systems. The analysis spans multiple architectures (GPT-2, LLaMA, Mistral) at scales from 70M to 12B parameters, with a temporal study across 154 Pythia training checkpoints, and produces open-source code, HuggingFace datasets, and interactive visualization dashboards.

### The Spin-Glass Picture

The self-attention mechanism for a single attention head can be identified as a mean-field update to the token representations: each token's new representation is a weighted average over all others, with weights set by a learned bilinear coupling $W_{QK} = W_Q^{T} W_K$. The multi-headed structure means that the expectation value appearing in this update takes the form of a *disordered average*, with the different heads encoding a frozen density of spin attractors specified by the trained weights.

This update, taken together with the residual stream connection, is structurally similar to the Thouless–Anderson–Palmer (TAP) equations for spin glasses, in which each spin's magnetization is obtained self-consistently as the thermal average of its couplings to all other spins. The TAP equations include an Onsager reaction term that removes the spurious self-interaction arising from a spin's influence on itself through the disordered medium; in its absence, the naive mean-field iteration formally admits an exponentially large number of fixed-point attractors that are nominally unstable. The transformer update lacks this correction, mirroring the naive mean-field form rather than its TAP counterpart and suggesting a regime characterized by a dense landscape of attractors. Two additional ingredients shape the dynamics: the low-rank structure of $W_Q$, $W_K$, and their product $W_{QK}$, which restricts the effective coupling to a small subspace of the residual stream; and the feedforward networks, which insert a specific non-linear evolution between iterations. Singular value decomposition and tools borrowed from resurgence in dynamical systems provide a natural language for analyzing both.

### Approach

The framework attacks this picture along four complementary axes:

- **Element-wise weight distributions** — Each attention head is treated as a frozen, out-of-equilibrium ensemble. Element-wise statistics and moments of the trained weights are compared to baseline expectations from random matrix theory to quantify how much, and in what direction, training has displaced the head from its initialization.

- **Spectral analysis** — Singular values of $W_{QK}$ and related operators are interpreted as energy-carrying modes of the system. The associated singular directions emerge from the bulk thermal distribution and carry physical meaning as learned features.

- **Cross-head correlations** — Heads are placed in relation to one another using both the similarity of their element distributions and a metric structure built from Frobenius distances between weight matrices. These correlations support the construction of an order parameter for the system as a whole.

- **Training evolution** — The same observables, tracked across training checkpoints, characterize the quenching process that effectively freezes the weights into their final non-equilibrium configuration.

Within individual models, different layers and attention heads exhibit distinct patterns of behavior. The animation below shows the architecture-wide evolution of the stable rank $r_s = \|W\|_F^2 / \|W\|_2^2$ across the Pythia training trajectory: a rapid early collapse onto a few leading directions in every (layer, head) cell, followed by a slower partial recovery. [Part 3](/posts/2026/transformer-training-dynamics/) unpacks this in terms of individual singular-value trajectories, in which modes emerge from the bulk, separate from one another, and stabilize as optimization proceeds.

<iframe src="/images/transformer-analysis/tp_architecture_evolution_stable_rank.html"
        width="100%" height="700" frameborder="0" loading="lazy"
        style="border:1px solid #ddd; border-radius:4px;"></iframe>
<p class="fig-caption">Architecture-wide evolution of stable rank $r_s = \|W\|_F^2 / \|W\|_2^2$ across training. The central heatmap shows $r_s$ for every (layer, head) at the selected training step; right and bottom marginals are the layer- and head-averaged values at that step. See <a href="/posts/2026/transformer-training-dynamics/">Part 3</a> for further discussion.</p>

[![Code](https://img.shields.io/badge/📁-Code_on_Github-blue)](https://github.com/angerami/transformer-analysis)
[![Dashboard](https://img.shields.io/badge/📊-Interactive_Dashboard-orange)](https://huggingface.co/spaces/angerami/transformer-weights)
[![Data](https://img.shields.io/badge/🗂️-Datasets_on_HuggingFace-yellow)](https://huggingface.co/collections/angerami/transformer-weight-evolution-study)

## Posts in This Series
{% assign series_posts = site.posts | where: "series", "transformer-spin" | sort: "series_order" %}
{% for post in series_posts %}
### Part {{ post.series_order }}: [{{ post.title }}]({{ post.url }})
*{{ post.date | date: "%B %d, %Y" }}*
{{ post.excerpt }}

---

{% endfor %}