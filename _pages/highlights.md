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
## Recognition & Media Coverage

### Landmark Discovery Recognition
**[*Observation of a Centrality-Dependent Dijet Asymmetry in Lead–Lead Collisions*](https://arxiv.org/abs/1011.6182)** - First direct observation of jet quenching:

* **Physical Review Letters Cover Article** (December 2010) with accompanying [Physics Viewpoint](https://physics.aps.org/articles/v3/105) commentary
* **CERN Press Release** - [*"LHC experiments bring new insight into primordial Universe"*](https://home.cern/news/press-release/cern/lhc-experiments-bring-new-insight-primordial-universe) highlighting ATLAS as "the first experiment to report direct observation of jet quenching"
* **CERN Courier Feature** - [*"ATLAS observes striking imbalance of jet energies in heavy-ion collisions"*](https://cerncourier.com/a/atlas-observes-striking-imbalance-of-jet-energies-in-heavy-ion-collisions/)

### Additional Research Recognition
* **DOE Office of Science Highlight** - [*"When in a Plasma of Quarks and Gluons, Not All Jets Radiate Equally"*](https://www.osti.gov/biblio/1871234) featuring subsequent jet substructure research

### Research Funding Recognition
* **DOE Artificial Intelligence Research Initiative** - Principal Investigator on competitive grant ([$20 million program announcement](https://www.energy.gov/articles/department-energy-announces-20-million-artificial-intelligence-research)) supporting [AI-Assisted Detector Design](/projects/) research

### Expert Commentary & Analysis
* **ATLAS Collaboration Briefings** - Author of research summaries for public communication:
  - [*"Quenching jets in hot, dense matter produced by colliding lead ions"*](https://atlas.cern/updates/briefing/quenching-jets-hot-dense-matter-produced-colliding-lead-ions)
  - [*"Study the QGP using muon pairs"*](https://atlas.cern/updates/briefing/study-QGP-using-muon-pairs)

---

*This page showcases recent activity across research, writing, and technical contributions. For complete information, see the dedicated sections linked above.*
