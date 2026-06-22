#!/usr/bin/env python3
"""
Generate comprehensive token usage table from all dataset feedback-loop runs.

Reads JSON refinement logs and generates:
1. Per-round tokens for each dataset
2. Cumulative tokens across 5 rounds
3. Saves as CSV and JSON for paper/analysis

Run after completing all feedback-loop reruns.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def extract_token_data(json_file):
    """Extract token data from refinement log JSON."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {json_file}: {e}")
        return None

def build_token_table(datasets_dir):
    """Build comprehensive token table from all datasets."""
    token_rows = []
    # Map dataset folder names to display names
    dataset_map = {
        '1-cic-iot': 'CIC-IOT',
        '2-wustl-iiot': 'WUSTL-IIOT',
        '3-ton-iot': 'TON-IOT',
        '4-bot-iot': 'BOT-IOT',
        '5-unsw-nb15': 'UNSW-NB15',
    }

    for folder_name, display_name in dataset_map.items():
        dataset_dir = Path(datasets_dir) / folder_name

        if not dataset_dir.exists():
            print(f"⚠ Directory not found: {dataset_dir}, skipping...")
            continue

        # Per-dataset results/ folders live outside the repo, in a sibling directory.
        results_dir = dataset_dir.parent.parent / "iot-llm-hids-pg-results" / folder_name / "llm"
        json_files = list(results_dir.glob("policy-refinement-summary-*.json")) if results_dir.exists() else []

        if not json_files:
            print(f"⚠ No refinement log found in {results_dir}, skipping {display_name}...")
            continue

        # Use the latest (or only) file
        json_file = sorted(json_files)[-1]
        print(f"✓ Reading {display_name}: {json_file}")

        data = extract_token_data(json_file)
        if not data:
            continue

        # Extract token info from rounds
        for round_data in data.get('rounds', []):
            row = {
                'Dataset': display_name,
                'Round': round_data['round'],
                'Macro F1': f"{round_data['macro_f1']:.4f}",
                'Round Prompt': round_data.get('round_prompt_tokens', 0),
                'Round Completion': round_data.get('round_completion_tokens', 0),
                'Round Total': round_data.get('round_total_tokens', 0),
                'Cumulative Prompt': round_data.get('cumulative_prompt_tokens', 0),
                'Cumulative Completion': round_data.get('cumulative_completion_tokens', 0),
                'Cumulative Total': round_data.get('cumulative_total_tokens', 0),
            }
            token_rows.append(row)

    return token_rows

def main():
    """Main execution."""
    # The script is located in iot-llm-hids-pg/, so datasets are direct siblings
    script_dir = Path(__file__).parent
    datasets_dir = script_dir  # Current directory contains 1-cic-iot, 2-wustl-iiot, etc.

    if not script_dir.exists():
        print(f"Error: Could not find script directory: {script_dir}")
        sys.exit(1)

    print("=" * 80)
    print("AGGREGATING TOKEN USAGE FROM ALL DATASETS")
    print(f"Looking in: {script_dir}")
    print("=" * 80 + "\n")

    # Build token table
    token_rows = build_token_table(datasets_dir)

    if not token_rows:
        print("Error: No token data found. Make sure you've run the feedback loops.")
        sys.exit(1)

    # Create DataFrame
    df = pd.DataFrame(token_rows)

    # Print formatted table
    print("\n" + "="*130)
    print("TOKEN USAGE SUMMARY - All Datasets, All Rounds")
    print("="*130)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("="*130)

    # Summary by dataset
    print("\nCUMULATIVE TOKEN USAGE BY DATASET (5-round total):")
    print("="*80)
    summary_rows = []
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        final_round = dataset_data[dataset_data['Round'] == 5].iloc[0] if len(dataset_data) > 0 else None
        if final_round is not None:
            summary_rows.append({
                'Dataset': dataset,
                'Total Prompt Tokens': int(final_round['Cumulative Prompt']),
                'Total Completion Tokens': int(final_round['Cumulative Completion']),
                'Total Tokens': int(final_round['Cumulative Total']),
            })

    summary_df = pd.DataFrame(summary_rows)
    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))
    print("="*80)

    # Save outputs
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Save full table as CSV
    csv_file = output_dir / "token_usage_all_rounds.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Full token table saved to: {csv_file}")

    # Save summary as JSON (for LaTeX table generation)
    summary_json = output_dir / "token_usage_summary.json"
    with open(summary_json, 'w') as f:
        json.dump({
            "description": "Token usage summary for policy refinement feedback loops",
            "model": "claude-haiku-4-5-20251001",
            "note": "Includes both per-round and cumulative token counts. Dollar cost omitted pending pricing documentation.",
            "cumulative_5_rounds": summary_df.to_dict('records'),
            "all_rounds": df.to_dict('records')
        }, f, indent=2)
    print(f"✓ Summary JSON saved to: {summary_json}")

    # Print note about pricing
    print("\n" + "="*80)
    print("NOTE ON PRICING:")
    print("="*80)
    print("Dollar cost NOT calculated. To include pricing, document:")
    print("  - Model name and version")
    print("  - Pricing date (effective date)")
    print("  - Input token price (per 1M tokens)")
    print("  - Output token price (per 1M tokens)")
    print("  - Whether cached tokens are counted separately")
    print("="*80)

if __name__ == '__main__':
    main()
