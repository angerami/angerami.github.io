---
layout: archive
title: "Transformer-Spin Correspondence"
permalink: /posts/transformer-spin/
author_profile: true
---

{% include base_path %}

## About This Series

This project treats transformer weights as a dynamical system, developing a statistical analysis pipeline that extracts interpretable metrics to track training evolution. The framework applies singular value decomposition, entropy, and KL divergence to characterize how attention mechanism weights evolve, revealing structure in the optimization process.
The analysis spans multiple architectures (GPT-2, LLaMA, Mistral) and scales (70M to 12B parameters), with temporal analysis across 154 Pythia training checkpoints. Within individual models, different layers and attention heads exhibit distinct patterns of behavior—and for Pythia, plotting singular values against training step reveals particle-like trajectories as modes emerge and stabilize during optimization. The work produces open-source code, HuggingFace datasets, and interactive visualization dashboards.

<!-- Figure placeholder: uncomment and update path/caption when ready
<figure>
  <img src="/images/transformer-analysis/FIGURE.png" alt="Series figure">
  <figcaption>Caption here.</figcaption>
</figure>
-->

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