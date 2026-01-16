---
layout: archive
title: "Projects • "
excerpt: "Research scientist applying experimental physics methodology to AI"
permalink: /projects/
author_profile: true
---

{% include base_path %}

# Projects

*My research spans nuclear and particle physics, machine learning applications, and the intersection of  physics with artificial intelligence. This work demonstrates how fundamental physics insights can drive innovations in AI, while advanced computational methods open new frontiers in experimental physics.*

---
# Generative AI: Research and Applied Projects
## The Physics of Transformers
**2024 - Present**  
*Independent Research*

This project treats transformer weights as a dynamical system, developing a statistical analysis pipeline that extracts interpretable metrics to track training evolution. The framework applies singular value decomposition, entropy, and KL divergence to characterize how attention mechanism weights evolve, revealing structure in the optimization process.
The analysis spans multiple architectures (GPT-2, LLaMA, Mistral) and scales (70M to 12B parameters), with temporal analysis across 154 Pythia training checkpoints. Within individual models, different layers and attention heads exhibit distinct patterns of behavior—and for Pythia, plotting singular values against training step reveals particle-like trajectories as modes emerge and stabilize during optimization. The work produces open-source code, HuggingFace datasets, and interactive visualization dashboards.

**Resources:**
<!-- - 📝 [Read the theory post](/posts/transformer-spin) *(coming soon)* -->
- 📝 [Read the post](/posts/2026/transformer-analysis) 
- 💻 [Code repository](https://github.com/angerami/transformer-analysis)
- 🎮 [Interactive explorer](https://huggingface.co/spaces/angerami/transformer-weights)


## Generative Modeling of Grateful Dead Setlists
**2025 - Present**  
*Independent Research*

This project builds generative models of concert setlists using supervised fine-tuning with GPT-2, treating the Grateful Dead's 30-year performance history as a natural language corpus. Each setlist becomes a sequence and each song a token, yielding a vocabulary of approximately 417 unique songs. Much like natural language, setlists exhibit patterns like opener conventions, set-closing sequences, and thematic pairings that experienced listeners recognize intuitively but that emerge statistically from the data.
The data pipeline initially processed Archive.org's 17,000+ concert recordings, providing an opportunity to develop fuzzy matching and vocabulary canonicalization techniques for messy real-world data. Production training uses cleaned data from setlist.fm for reliability. The availability of original recordings on Archive.org offers a compelling path to extend this work into audio modality, connecting setlist structure to the musical content itself.

**Resources:**
- 💻 [Code repository](https://github.com/angerami/dead-setlist)
<!-- - 📝 [Read the post](/posts/dead-setlist) *(coming soon)* 
- 🎮 [Demo](https://huggingface.co/spaces/angerami/XXX) -->

**Skills:** 
<span class="skill-tag">SFT</span>
<span class="skill-tag">NLP</span>
<span class="skill-tag">Gen AI</span>
<span class="skill-tag">LLM</span>

## Generative Modeling of Baseball Game States
**2025 - Present**  
*Independent Research*

This project applies transformer language models to sequential game state prediction, treating baseball games as sequences of discrete state transitions. The system processes 3.3 million pitch sequences from MLB's Statcast (2015–present) and Retrosheet's historical archives (1871–present), converting games into tokenized sequences for language model training.
The research addresses a question in mechanistic interpretability: how do traditional baseball statistics emerge as learned properties from underlying game dynamics? The evaluation framework assesses not just prediction accuracy but whether models learn actual baseball rules—detecting illegal transitions that violate game constraints. State representations range from simple 24-state models encoding outs and baserunners to complex 57,000-state encodings that capture detailed game context.

**Resources:**
- 💻 [Code repository](https://github.com/angerami/baseball-states)

**Skills:** 
<span class="skill-tag">Pretraining</span>
<span class="skill-tag">NLP</span>
<span class="skill-tag">Gen AI</span>
<span class="skill-tag">LLM</span>

---

## Machine Learning & AI Applications in Physics

### AI-Informed Detector Design
**Jan 2021 - Jan 2024**  
*Lawrence Livermore National Laboratory*

Using deep learning as a new tool to guide the design of detectors for collider physics experiments. This work represents a paradigm shift from traditional engineering approaches to ML-optimized detector configurations.

**Key Achievements:**
- Improved energy resolution by 40% over baseline designs
- Established optimization framework for future detector systems
- Demonstrated 50% improvement in particle identification accuracy

**Skills:** <span class="skill-tag">Machine Learning</span><span class="skill-tag">Deep Learning</span><span class="skill-tag">Generative AI Tools</span><span class="skill-tag">Experimental Physics</span><span class="skill-tag">Data Analysis</span>

**Publications:**
- [Design of a SiPM-on-Tile ZDC for the future EIC and its Performance with Graph Neural Networks](https://arxiv.org/abs/2406.12877)
- [The Optimal use of Segmentation for Sampling Calorimeters](https://arxiv.org/abs/2310.04442)
- [Comparison of Point Cloud and Image-based Models for Calorimeter Fast Simulation](https://arxiv.org/abs/2307.04780)

### Improving Particle Reconstruction with Deep Learning
**Jan 2019 - Jan 2022**  
*Lawrence Livermore National Laboratory · ATLAS Experiment at CERN*

We utilized deep learning methods to improve particle identification and reconstruction using the ATLAS calorimeter. We studied convolutional, graph and transformer architectures and compared their results, achieving major improvements in energy calibration by using full spatial information from electromagnetic and hadronic showers.

**Key Achievements:**
- 50% improvement in energy resolution
- 10x reduction in false positive rate while maintaining >95% recall
- Demonstrated superiority of graph neural networks for particle reconstruction

**Skills:** <span class="skill-tag">Experimental Physics</span><span class="skill-tag">Data Analysis</span><span class="skill-tag">Machine Learning</span>

**Code Repositories:**
- [GitHub - atlas-calo-ml/MLTree](https://github.com/atlas-calo-ml/MLTree)
- [GitHub - atlas-calo-ml/gn4pions_eastbay: Using graph_nets for pion classification and energy](https://github.com/atlas-calo-ml/gn4pions_eastbay)

**Publications:**
- [Point Cloud Deep Learning Methods for Pion Reconstruction in the ATLAS Experiment](https://cds.cern.ch/record/2825379)

---

## Heavy-Ion Physics & QCD

### Jet Energy Loss and Substructure
**Jan 2019 - Jan 2022**  
*Lawrence Livermore National Laboratory · ATLAS Experiment at CERN*

*Can we experimentally observe whether wide jets lose more energy than narrow ones?*

This project explored fundamental questions about how jets interact with the quark-gluon plasma, providing new insights into the substructure dependence of energy loss mechanisms.

**Skills:** <span class="skill-tag">Experimental Physics</span><span class="skill-tag">Data Analysis</span><span class="skill-tag">Research Projects</span><span class="skill-tag">Analytical Skills</span><span class="skill-tag">Uncertainty Quantification</span>

**Publications:**
- [Measurement of substructure-dependent jet suppression in Pb+Pb collisions at 5.02 TeV with the ATLAS detector](https://arxiv.org/abs/2211.11470)

**Recognition:**
- [DOE Office of Science Highlight: "When in a Plasma of Quarks and Gluons, Not All Jets Radiate Equally"](https://www.osti.gov/biblio/1871234)

### New Approaches to Ultra-Peripheral Collisions
*Lawrence Livermore National Laboratory · ATLAS Experiment at CERN*

Advanced studies of ultra-peripheral heavy-ion collisions, exploring photonuclear processes and novel QCD phenomena in extreme electromagnetic field environments.

**Skills:** <span class="skill-tag">Experimental Physics</span><span class="skill-tag">Data Analysis</span><span class="skill-tag">Analytical Skills</span><span class="skill-tag">Uncertainty Quantification</span><span class="skill-tag">Monte Carlo Simulation</span><span class="skill-tag">High Performance Computing</span>

**Publications:**
- [Measurement of photonuclear jet production in ultra-peripheral Pb+Pb collisions](https://arxiv.org/abs/2409.11060)
- [Observation of centrality-dependent acoplanarity for muon pairs produced via two-photon scattering](https://arxiv.org/abs/1806.08708)

### Observation of Jet Quenching at the LHC
*Columbia University · ATLAS Experiment at CERN*

*Landmark discovery that ushered in the LHC era of heavy-ion physics*

In the first Pb+Pb collisions at the LHC, the ATLAS experiment observed highly imbalanced dijet pairs, providing the first direct evidence of jet quenching in the quark-gluon plasma at unprecedented energies.

**Skills:** <span class="skill-tag">Independent Research</span><span class="skill-tag">Data Analysis</span>

**Publications:**
- **[Observation of a Centrality-Dependent Dijet Asymmetry in Lead-Lead Collisions](https://arxiv.org/abs/1011.6182)** *(First jet measurement in heavy-ion collisions - field-defining result)*

### Precision Measurements of Jet Quenching
*Columbia University · ATLAS Experiment at CERN*

Systematic studies of jet suppression phenomena, establishing quantitative frameworks for understanding energy loss in the quark-gluon plasma.

**Skills:** <span class="skill-tag">Experimental Physics</span><span class="skill-tag">Data Analysis</span><span class="skill-tag">Uncertainty Quantification</span><span class="skill-tag">Monte Carlo Simulation</span><span class="skill-tag">Analytical Skills</span>

**Publications:**
- [Measurement of jet pT correlations in Pb+Pb and pp collisions](https://arxiv.org/abs/1706.09363)
- [Measurements of the Nuclear Modification Factor for Jets in Pb+Pb Collisions](https://arxiv.org/abs/1411.2357)
- [Jet size dependence of single jet suppression in lead-lead collisions](https://arxiv.org/abs/1208.1967)

---