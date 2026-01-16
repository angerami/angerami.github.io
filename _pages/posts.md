---
layout: archive
title: "Writing"
permalink: /posts/
author_profile: true
---

{% include base_path %}

*Technical explorations at the intersection of physics and artificial intelligence*

<!-- ---

# Series

## [The Transformer-Spin Correspondence](/posts/transformer-spin/)
Exploring the deep mathematical connections between transformer neural networks and spin glass systems from statistical physics. This series reveals how concepts from statistical physics illuminate the inner workings of large language models.

--- -->

# Recent Posts

{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}

{% if site.posts.size == 0 %}
*Posts coming soon. In-depth explorations of statistical physics, AI architectures, and research methodologies.*
{% endif %}