#!/usr/bin/env python3
"""
Fix token aggregation: normalises both key families and writes corrected CSV.

CIC-IoT  uses: round_prompt_tokens / round_completion_tokens / round_total_tokens
WUSTL    uses: prompt_tokens / completion_tokens / total_tokens

Both are normalised to: prompt_tokens, completion_tokens, total_tokens
Cumulative totals are computed by summing across all five rounds.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent
# Per-dataset results/ folders live outside the repo, in a sibling directory.
EXTERNAL_RESULTS_ROOT = BASE.parent / "iot-llm-hids-pg-results"

DATASETS = [
    ("1-cic-iot",    "CIC-IoT2023"),
    ("2-wustl-iiot", "WUSTL-IIoT"),
    ("3-ton-iot",    "TON-IoT"),
    ("4-bot-iot",    "Bot-IoT"),
    ("5-unsw-nb15",  "UNSW-NB15"),
]

def extract_round_tokens(round_data):
    """Return (prompt, completion, total) regardless of key family."""
    if "round_prompt_tokens" in round_data:
        return (
            round_data["round_prompt_tokens"],
            round_data["round_completion_tokens"],
            round_data["round_total_tokens"],
        )
    return (
        round_data.get("prompt_tokens", 0),
        round_data.get("completion_tokens", 0),
        round_data.get("total_tokens", 0),
    )

rows = []

for folder, label in DATASETS:
    llm_dir = EXTERNAL_RESULTS_ROOT / folder / "llm"
    if not llm_dir.exists():
        print(f"  SKIP {label}: directory not found")
        continue

    jsons = sorted(llm_dir.glob("policy-refinement-summary-*.json"))
    if not jsons:
        print(f"  SKIP {label}: no policy-refinement-summary JSON")
        continue

    path = jsons[-1]
    with open(path) as f:
        data = json.load(f)

    prompt_total = completion_total = total_total = 0
    for rd in data.get("rounds", []):
        p, c, t = extract_round_tokens(rd)
        prompt_total     += p
        completion_total += c
        total_total      += t

    rows.append({
        "dataset":           label,
        "prompt_tokens":     prompt_total,
        "completion_tokens": completion_total,
        "total_tokens":      total_total,
    })
    print(f"  OK  {label}: prompt={prompt_total:,}  completion={completion_total:,}  total={total_total:,}  (from {path.name})")

# Write CSV
out = BASE / "results" / "token_usage_corrected.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    f.write("dataset,prompt_tokens,completion_tokens,total_tokens\n")
    for r in rows:
        f.write(f"{r['dataset']},{r['prompt_tokens']},{r['completion_tokens']},{r['total_tokens']}\n")

# Print table
print()
print("=" * 65)
print(f"{'Dataset':<15} {'Prompt':>12} {'Completion':>12} {'Total':>12}")
print("-" * 65)
for r in rows:
    print(f"{r['dataset']:<15} {r['prompt_tokens']:>12,} {r['completion_tokens']:>12,} {r['total_tokens']:>12,}")
print("=" * 65)
print(f"\nSaved to: {out}")
