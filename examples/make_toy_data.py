#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a tiny toy dataset that mimics the real pipeline formats.

Running this writes:
  * examples/toy.en-de.gen.json   -- like generate.py output (2 systems x 3 segments x 3 candidates)
  * examples/toy_human.en-de.json -- like data/wmt24/2024_en-de.json (human error spans)

It lets you smoke-test rerank.py + evaluate.py end-to-end without a GPU or the
full WMT data. See examples/README.md.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def ann(*spans):
    """Build a raw model annotation JSON string from (text, category, sub, severity)."""
    errors = [{"error_span": t, "category": c, "sub_category": s, "severity": sev}
              for (t, c, s, sev) in spans]
    return json.dumps({"errors": errors}, ensure_ascii=False)


# Three source/translation pairs shared by both toy "systems".
SEGMENTS = [
    {"src": "The cat sat on the mat.",   "hypo": "Die Katze saß auf dem Mond."},   # 'Mond' = wrong (mistranslation)
    {"src": "She sells seashells.",       "hypo": "Sie verkauft Muscheln."},        # correct
    {"src": "It was a dark night.",       "hypo": "Es war eine dunkle nacht ."},     # 'nacht' lowercase + spaced '.'
]

# Per system, per segment: a list of (annotation_json, logprob) candidates.
GEN = {
    "SystemA": [
        [(ann(("Mond", "Accuracy", "Mistranslation", "major")), -2.0),
         (ann(("Mond", "Accuracy", "Mistranslation", "major"),
              ("saß", "Style", "Awkward", "minor")), -3.1),
         (ann(), -5.0)],
        [(ann(), -1.0),
         (ann(("Muscheln", "Accuracy", "Mistranslation", "minor")), -4.2),
         (ann(), -1.5)],
        [(ann(("nacht", "Fluency", "Spelling", "minor")), -2.2),
         (ann(("nacht", "Fluency", "Spelling", "minor"),
              (" .", "Fluency", "Punctuation", "minor")), -2.9),
         (ann(("dunkle nacht", "Accuracy", "Mistranslation", "major")), -6.0)],
    ],
    "SystemB": [
        [(ann(("auf dem Mond", "Accuracy", "Mistranslation", "major")), -2.5),
         (ann(("Mond", "Accuracy", "Mistranslation", "major")), -2.6),
         (ann(("Katze", "Accuracy", "Mistranslation", "major")), -7.0)],
        [(ann(), -0.8),
         (ann(), -0.9),
         (ann(("Sie", "Accuracy", "Addition", "minor")), -5.5)],
        [(ann(("nacht", "Fluency", "Spelling", "minor")), -1.9),
         (ann(), -3.0),
         (ann(("nacht", "Fluency", "Spelling", "minor")), -2.0)],
    ],
}

# Human gold error spans (character offsets into the translation).
HUMAN = {
    # segment index -> list of {severity, error_span, start, end, category}
    0: [{"severity": "major", "error_span": "Mond", "start": 23, "end": 27,
         "category": "Accuracy/Mistranslation"}],
    1: [],  # no error
    2: [{"severity": "minor", "error_span": "nacht", "start": 18, "end": 23,
         "category": "Fluency/Spelling"},
        {"severity": "minor", "error_span": " .", "start": 23, "end": 25,
         "category": "Fluency/Punctuation"}],
}


def main():
    # gen.json : {system: [[[text, logp, hypo, src], ...], ...]}
    gen_out = {}
    for system, seg_cands in GEN.items():
        segments = []
        for seg_idx, cands in enumerate(seg_cands):
            hypo = SEGMENTS[seg_idx]["hypo"]
            src = SEGMENTS[seg_idx]["src"]
            # sort best-logprob-first, as generate.py does
            cands = sorted(cands, key=lambda x: x[1], reverse=True)
            segments.append([[text, logp, hypo, src] for text, logp in cands])
        gen_out[system] = segments
    gen_path = os.path.join(HERE, "toy.en-de.gen.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_out, f, ensure_ascii=False, indent=2)
    print("wrote", gen_path)

    # human file : [{system, source, target, error_span:[{severity,start,end,...}]}]
    human_rows = []
    for system in GEN:
        for seg_idx, seg in enumerate(SEGMENTS):
            spans = [{"severity": e["severity"],
                      "category": e["category"],
                      "error_span": e["error_span"],
                      "error_span_start": e["start"],
                      "error_span_end": e["end"]}
                     for e in HUMAN[seg_idx]]
            human_rows.append({
                "system": system,
                "source": seg["src"],
                "target": seg["hypo"],
                "error_span": spans,
            })
    human_path = os.path.join(HERE, "toy_human.en-de.json")
    with open(human_path, "w", encoding="utf-8") as f:
        json.dump(human_rows, f, ensure_ascii=False, indent=2)
    print("wrote", human_path)


if __name__ == "__main__":
    main()
