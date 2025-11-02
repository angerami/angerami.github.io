---
layout: archive
title: "Transformer-Spin Correspondence"
permalink: /posts/transformer-spin/
author_profile: true
---

{% include base_path %}

## About This Series
[Description]

## Posts in This Series
{% assign series_posts = site.posts | where: "series", "transformer-spin" | sort: "series_order" %}
{% for post in series_posts %}
### Part {{ post.series_order }}: [{{ post.title }}]({{ post.url }})
*{{ post.date | date: "%B %d, %Y" }}*
{{ post.excerpt }}

---

{% endfor %}