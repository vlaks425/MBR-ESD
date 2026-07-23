# Data

## Human error spans (included)

`wmt24/2024_<lp>.json` — human MQM error-span annotations for the WMT24 Metrics
Shared Task, for `lp ∈ {en-de, en-es, ja-zh}`. These are consumed by
[`evaluate.py`](../evaluate.py) as the gold reference for span-level F1 / SoftF1.

Each file is a JSON array of segment objects:

```json
{
  "system": "Aya23",
  "source": "…source segment…",
  "target": "…machine translation…",
  "error_span": [
    {"severity": "major", "category": "Accuracy/Mistranslation",
     "error_span": "…", "error_span_start": 23, "error_span_end": 27}
  ]
}
```

Predictions are matched to a human segment by `system` and `source + target`.
These annotations are derived from the public WMT24 MQM data; please also cite
the WMT24 Metrics Shared Task if you use them.

> ESA (rather than MQM) span annotations were only available for `en-es` in our
> setup, so the pipeline here targets the MQM annotations for all three pairs.

## WMT source / system outputs / references (download separately)

`generate.py` and the system/segment-level meta-evaluation need the WMT source
segments and the system translations (hypotheses). These are part of the
**mt-metrics-eval-v2** release and are **not redistributed here**. Download them
from the official toolkit:

- https://github.com/google-research/mt-metrics-eval

After downloading, point the pipeline at your local copy (default location
`data/mt-metrics-eval-v2`, or set `MTME_ROOT`):

```
data/mt-metrics-eval-v2/wmt24/
├── sources/<lp>.txt              # one source segment per line
├── system-outputs/<lp>/*.txt     # one file per system, one translation per line
├── references/<lp>.refA.txt
└── human-scores/<lp>.mqm.seg.score
```
