---
layout: archive
title: "Recent Highlights"
permalink: /highlights/
author_profile: true
---

{% include base_path %}

*Latest writing, research, and technical contributions across platforms*

---

## Recent Writing

### Medium Articles
* **[Coming Soon: The Physics Hidden in Your AI](https://medium.com/@aaron.angerami)** - *Exploring the transformer-spin correspondence*  
  *Published on Medium*

### Technical Posts
{% for post in site.posts limit:3 %}
* **[{{ post.title }}]({{ post.url }})** - {{ post.excerpt | strip_html | truncatewords: 20 }}  
  *{{ post.date | date: "%B %Y" }}*
{% endfor %}

{% if site.posts.size == 0 %}
* **Technical deep-dives coming soon** - In-depth explorations of statistical physics, AI architectures, and research methodologies
{% endif %}

---

## Recent Publications

### Latest Research
* ATLAS Collaboration, *Measurement of photonuclear jet production in ultra-peripheral Pb+Pb collisions at √sNN=5.02 TeV with the ATLAS detector*, Submitted to Phys. Rev. D. (2024), [arXiv:2409.11060](https://arxiv.org/abs/2409.11060)

* R. Milton et al., *Design of a SiPM-on-Tile ZDC for the future EIC and its Performance with Graph Neural Networks*, Submitted to JINST (2024), [arXiv:2406.12877](https://arxiv.org/abs/2406.12877)

[View all publications →](/publications/)

---

## Recent Talks

### 2023-2024 Highlights
* **ATLAS Overview** - Quark Matter 2025, Houston, TX (September 2023)
* **Applications of AI and ML to Nuclear Physics** - National Nuclear Physics Summer School (July 2023)
* **Machine Learning Approaches to Calorimetric Particle Reconstruction** - University of Washington (August 2022)

[View complete speaking history →](/talks/)

---

## Current Projects

### Active Research
* **Transformer-Spin Correspondence** - Developing mathematical connections between neural attention mechanisms and statistical physics
* **AI-Optimized Detector Design** - Using machine learning to guide next-generation particle detector configurations
* **Physics-Informed Machine Learning** - Applying domain knowledge to improve ML model performance and interpretability

[View project portfolio →](/projects/)

---

*This page showcases recent activity across research, writing, and technical contributions. For complete information, see the dedicated sections linked above.*
