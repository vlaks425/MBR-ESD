#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate error-span annotations (candidates) for Error Span Detection with vLLM.

Given the WMT system translations (hypotheses) and their source segments, this
script prompts an ESD model to produce, for every (source, translation) pair,
one or more error-span annotations in JSON. These candidates are then reranked
by ``rerank.py`` (MAP or one of the MBR utilities) and scored by ``evaluate.py``.

Two decoding regimes, matching the paper:

* **MAP** (greedy, ``--num_hypotheses 1 --temperature 0``): a single annotation.
  This is what you run for the distilled model, whose single greedy output is
  trained to approximate the MBR decision.
* **MBR** (sampling, e.g. ``--num_hypotheses 256 --temperature 2.0 --top_k 10``):
  many candidate annotations per segment, later reranked by ``rerank.py``.

Output
------
A JSON file ``<output_prefix>.<lang_pair>.gen.json`` shaped as::

    {
      "<system_name>": [
        [ [candidate_text, cumulative_logprob, translation, source], ... ],  # segment 0, N candidates
        ...
      ],
      ...
    }

Reference-free (QE) prompting: the reference translation is never shown to the
model; only the source and the machine translation are.
"""

import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The instruction shown to the model. It defines the MQM-style error taxonomy and
# forces a JSON list of {error_span, category, sub_category, severity} objects.
PROMPT_TEMPLATE = """Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, whitespace,capitalization, register, spelling, unnatural flow), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
    Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.

    Please output in JSON format where each error span contains the following fields: error_span, category, sub_category, severity.
    If no errors are found, output an empty list `[]`
    Please do not output anything else.

    {source_lang} source:
    ```{source_seg}```
    {target_lang} translation:
    ```{target_seg}```"""

# ISO code -> language name, used to fill the prompt template.
LANG_CODE_TO_NAME = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "he": "Hebrew",
    "ru": "Russian", "ro": "Romanian", "cs": "Czech", "uk": "Ukrainian",
    "is": "Icelandic", "hi": "Hindi",
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate ESD candidates with vLLM.")
    p.add_argument("--model_path", required=True,
                   help="HF model id or local path (base or distilled ESD model).")
    p.add_argument("--src_file", required=True,
                   help="Source file; one segment per line. Named '<src>-<tgt>.txt'.")
    p.add_argument("--hypo_dir", required=True,
                   help="Directory of system-output files (one file per system, "
                        "one translation per line, aligned to --src_file).")
    p.add_argument("--lang_pair", required=True, help="e.g. 'en-de'.")
    p.add_argument("--output_prefix", required=True,
                   help="Output written to '<output_prefix>.<lang_pair>.gen.json'.")
    p.add_argument("--num_hypotheses", type=int, default=1,
                   help="Candidates sampled per segment (1 = MAP/greedy, >1 = MBR).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy; paper uses 2.0 for MBR).")
    p.add_argument("--top_k", type=int, default=0, help="Top-k (0 disables).")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus).")
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=2048,
                   help="Max new tokens per candidate.")
    p.add_argument("--gpu", type=str, default="0",
                   help="Comma-separated GPU ids; count sets tensor_parallel_size.")
    return p.parse_args()


def build_prompt(tokenizer, source_lang, target_lang, source_seg, target_seg):
    """Render the chat prompt for one (source, translation) pair."""
    prompt = PROMPT_TEMPLATE.format(
        source_lang=source_lang, target_lang=target_lang,
        source_seg=source_seg.strip(), target_seg=target_seg.strip(),
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    # vLLM adds the BOS itself, so strip a leading BOS from the template.
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return text


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from pydantic import BaseModel
    from tqdm import tqdm

    # Constrain generation to the exact JSON schema of an error-span list. This
    # guarantees parseable output (guided/structured decoding).
    class SpanError(BaseModel):
        error_span: str
        category: str
        sub_category: str
        severity: str

    class Errors(BaseModel):
        errors: list[SpanError]

    guided_decoding = GuidedDecodingParams(json=Errors.model_json_schema())
    sampling_params = SamplingParams(
        n=args.num_hypotheses,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        logprobs=0,               # keep cumulative_logprob for MAP reranking
        guided_decoding=guided_decoding,
    )

    src_code, tgt_code = args.lang_pair.split("-")
    source_lang = LANG_CODE_TO_NAME[src_code]
    target_lang = LANG_CODE_TO_NAME[tgt_code]

    with open(args.src_file, "r", encoding="utf-8") as f:
        src_lines = [line.rstrip("\n") for line in f]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, add_eos_token=False, add_bos_token=False, padding_side="left")

    n_gpus = len(args.gpu.split(","))
    print(f"Loading {args.model_path} on {n_gpus} GPU(s)...")
    model = LLM(
        model=args.model_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=n_gpus,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        max_num_batched_tokens=8196,
    )

    results = {}
    for hypo_file in tqdm(sorted(os.listdir(args.hypo_dir)), desc="Systems", unit="sys"):
        system_name = os.path.splitext(hypo_file)[0]
        with open(os.path.join(args.hypo_dir, hypo_file), "r", encoding="utf-8") as f:
            hypo_lines = [line.rstrip("\n") for line in f]
        assert len(hypo_lines) == len(src_lines), (
            f"{hypo_file}: {len(hypo_lines)} translations != {len(src_lines)} sources")

        prompts = [
            build_prompt(tokenizer, source_lang, target_lang, s, h)
            for s, h in zip(src_lines, hypo_lines)
        ]
        outputs = model.generate(prompts, sampling_params=sampling_params)

        # For each segment, keep all N candidates sorted by logprob (best first).
        segments = []
        for src, hypo, output in zip(src_lines, hypo_lines, outputs):
            cands = [(seq.text, seq.cumulative_logprob) for seq in output.outputs]
            cands.sort(key=lambda x: x[1], reverse=True)
            assert len(cands) == args.num_hypotheses
            segments.append([[text, logp, hypo.strip(), src.strip()]
                             for text, logp in cands])
        results[system_name] = segments

    out_path = f"{args.output_prefix}.{args.lang_pair}.gen.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
