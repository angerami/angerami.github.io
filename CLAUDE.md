# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Local development
bundle exec jekyll serve          # Serve with live reload at localhost:4000
docker-compose up                 # Alternative: serve via Docker

# Build
bundle exec jekyll build          # Production build

# JavaScript assets
npm run watch:js                  # Watch and rebuild JS on changes
npm run build:js                  # Minify JS for production

# CV data pipeline
./scripts/update_cv_json.sh       # Regenerate _data/cv.json from _pages/cv.md

# Setup (first time)
bundle install
npm install
```

## Architecture

This is an academic portfolio/blog site for Aaron Angerami (research scientist, transformers & physics). Built on Jekyll with a customized Minimal Mistakes theme.

**Content model:**
- `_posts/` — Blog posts. Currently a series on transformer analysis and physics ("transformer-spin" series). Markdown with YAML frontmatter.
- `_pages/` — Static pages (about, CV, projects, presentations, etc.). Served at permalink URLs defined in frontmatter.
- `_drafts/` — Unpublished posts; visible when serving with `--drafts` flag.
- `_data/cv.json` — Structured CV data consumed by `cv-layout.html`. Generated from `_pages/cv.md` via `scripts/cv_markdown_to_json.py` — edit the markdown, then regenerate.
- `_data/navigation.yml` — Site navigation links.
- `images/` — Post assets organized in subdirectories by topic.

**Layouts and templates** (`_layouts/`, `_includes/`):
- `single.html` — Primary template for posts and pages.
- `talk.html` — Specialized layout for conference talks.
- `cv-layout.html` — Renders CV from `_data/cv.json`.
- `_includes/` — 39 reusable components (author sidebar, nav, analytics, etc.).

**Styling** (`_sass/`):
- Custom overrides live in `_sass/custom/`: `_custom.scss`, `_syntax.scss`, `_themes.scss`.
- Theme vendor SCSS is in `_sass/minimal-mistakes/`.

**JS pipeline:** Source JS in `assets/js/` → minified output in `assets/js/main.min.js`. jQuery-based with plugins: FitVids (responsive video), Magnific Popup (lightbox), jQuery Smooth Scroll.

**CI/CD:** GitHub Actions (`.github/workflows/jekyll.yml`) builds and deploys to GitHub Pages on push to `master`. A second workflow (`scrape_talks.yml`) automates talk data extraction.

## Jekyll Plugins in Use

`jekyll-feed`, `jekyll-sitemap`, `jekyll-paginate`, `jekyll-redirect-from`, `jekyll-gist`, `jemoji`

## Useful Patterns

- Posts use a `series` frontmatter key to group multi-part content.
- Draft/WIP posts display a yellow warning banner (configured in frontmatter).
- The talkmap (`talkmap/`) is a Python/Jupyter notebook that generates an interactive conference map hosted externally on HuggingFace Spaces.
