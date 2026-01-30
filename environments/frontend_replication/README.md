# frontend-replication

### Overview
- **Environment ID**: `frontend-replication`
- **Short description**: Multi-turn benchmark where models replicate real websites from screenshots and natural language descriptions using HTML/CSS/JS. Scores visual fidelity using Design2Code metrics (block-match, text, position, color, CLIP).
- **Tags**: frontend, multi-turn, long-horizon, benchmark, eval, design2code

### Datasets
- **Primary dataset(s)**: Built-in pilot dataset (5 websites), expandable to 100 curated tasks
- **Source links**: Manually curated from public websites (brex.com, stripe.com, linear.app, vercel.com, notion.so)
- **Split sizes**: Eval-only (benchmark)

### Task
- **Type**: multi-turn (up to 20 turns)
- **Parser**: HTML extraction from ```html code blocks
- **Rubric overview**: Five equally-weighted Design2Code metrics (each 0–1):
  - **Block-Match (Size)**: Hungarian matching of visual blocks by bounding box area
  - **Text**: SequenceMatcher on matched block text content
  - **Position**: 1 - Chebyshev distance between matched block positions
  - **Color**: CIEDE2000 delta-E between matched block colors
  - **CLIP**: Cosine similarity of CLIP embeddings on inpainted screenshots

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run frontend-replication
```

Configure model and sampling:

```bash
prime eval run frontend-replication \
  -m gpt-4.1-mini \
  -n 5 -r 1 -t 4096 -T 0.7 \
  -a '{"max_turns": 20}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Requires Playwright and Chromium installed (`playwright install chromium`).
- CLIP scoring requires `transformers` and `torch` (install with `pip install frontend-replication[clip]`).

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str \| None | `None` | HuggingFace dataset path or local JSON (None = pilot dataset) |
| `num_eval_examples` | int | `-1` | Number of eval examples (-1 for all) |
| `max_turns` | int | `20` | Maximum turns per rollout |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Final Design2Code visual similarity score (average of 5 components) |
| `size_score_metric` | Block-match size similarity |
| `text_score_metric` | Matched block text similarity |
| `position_score_metric` | Matched block position similarity |
| `color_score_metric` | CIEDE2000 color similarity |
| `clip_score_metric` | CLIP visual similarity (with inpainting) |
| `turns_used` | Number of turns the model used |
| `avg_render_time` | Average HTML rendering time per turn (seconds) |
| `did_signal_done` | Whether the model signaled completion (1.0 or 0.0) |

### How It Works
1. Model receives a reference screenshot + natural language description of a target website
2. Model writes HTML/CSS/JS in a ```html code block
3. Environment renders the HTML via headless Chromium (Playwright)
4. Model receives its rendered screenshot alongside the reference for comparison
5. Model refines iteratively until it signals DONE or hits max_turns
6. Final rendering is scored against reference using Design2Code metrics

Images and logos are handled via inpainting — the model should use colored placeholder divs with correct dimensions. The scoring pipeline inpaints detected image regions before CLIP comparison, so models are not penalized for using placeholders.
