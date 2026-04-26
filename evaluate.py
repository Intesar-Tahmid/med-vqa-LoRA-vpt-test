#!/usr/bin/env python3
"""
Per-category evaluation: Accuracy, BERTScore, LAVE.
Matches the evaluation protocol in BanglaMedVQA Tables 4 & 5.

Requires:
    pip install bert-score openai

Usage:
    export OPENAI_API_KEY=your_key
    python evaluate.py \\
        --predictions_csv output/A4/chest_x-ray_test_predictions.csv \\
        --output_json results/A4_results.json

    # Skip LAVE (no OpenAI key needed):
    python evaluate.py --predictions_csv output/A4/... --no_lave
"""

import os
import json
import argparse
import pandas as pd

CATEGORIES = ["modality", "organ", "abnormality", "condition", "position"]
LAVE_BATCH  = 20


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["match"] = (
        df["predicted_answer_bn"].str.strip().str.lower() ==
        df["llm_answer_bn"].str.strip().str.lower()
    )
    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["match"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["match"].mean() * 100, 2)
    return results


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(df: pd.DataFrame) -> dict:
    """BERTScore F1 with xlm-roberta-large (multilingual, handles Bangla)."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError("Run: pip install bert-score>=0.3.13")

    preds = df["predicted_answer_bn"].fillna("").tolist()
    refs  = df["llm_answer_bn"].fillna("").tolist()

    _, _, F1 = bert_score_fn(preds, refs, lang="bn", verbose=False)
    df = df.copy()
    df["bscore"] = F1.numpy()

    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["bscore"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["bscore"].mean() * 100, 2)
    return results


# ---------------------------------------------------------------------------
# LAVE
# ---------------------------------------------------------------------------

def compute_lave(df: pd.DataFrame, openai_client) -> dict:
    """
    LLM-Assisted VQA Evaluation using GPT-4.1-mini as judge.
    Matches the LAVE protocol from Mañas et al. 2024.
    """
    preds = df["predicted_answer_bn"].fillna("").tolist()
    refs  = df["llm_answer_bn"].fillna("").tolist()
    scores = []

    for i in range(0, len(preds), LAVE_BATCH):
        batch_preds = preds[i:i + LAVE_BATCH]
        batch_refs  = refs[i:i + LAVE_BATCH]

        pairs = "\n".join([
            f"{j+1}. Reference: {r} | Prediction: {p}"
            for j, (r, p) in enumerate(zip(batch_refs, batch_preds))
        ])

        prompt = (
            "You are a strict evaluator for a medical VQA task in Bangla.\n"
            "For each pair below, output ONLY a JSON list of floats in [0,1].\n"
            "Score 1.0 if the prediction is semantically equivalent to the reference, "
            "0.5 if partially correct, 0.0 if wrong or missing.\n"
            "Do NOT output any text outside the JSON list.\n\n"
            f"{pairs}"
        )

        try:
            response = openai_client.chat.completions.create(
                model       = "gpt-4.1-mini",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0,
            )
            batch_scores = json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"  Warning: LAVE batch {i//LAVE_BATCH + 1} failed ({e}). Using 0.")
            batch_scores = [0.0] * len(batch_preds)

        scores.extend(batch_scores)

    df = df.copy()
    df["lave"] = scores[:len(df)]

    results = {}
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        results[cat] = round(sub["lave"].mean() * 100, 2) if len(sub) > 0 else 0.0
    results["overall"] = round(df["lave"].mean() * 100, 2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-category evaluation: Accuracy, BERTScore, LAVE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_csv", required=True,
                        help="CSV from run_real_inference() — must have "
                             "predicted_answer_bn, llm_answer_bn, category columns")
    parser.add_argument("--output_json",     default="evaluation_results.json")
    parser.add_argument("--no_lave",         action="store_true", default=False,
                        help="Skip LAVE (no OpenAI key required)")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv)

    required = ["predicted_answer_bn", "llm_answer_bn", "category"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Make sure train_model.py populates the 'category' column in the CSV."
        )

    print(f"Evaluating {len(df)} samples ...")

    print("\nComputing Accuracy ...")
    acc = compute_accuracy(df)

    print("Computing BERTScore ...")
    bsc = compute_bertscore(df)

    lave = None
    if not args.no_lave:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set — skipping LAVE. Use --no_lave to suppress this message.")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                print("Computing LAVE (GPT-4.1-mini) ...")
                lave = compute_lave(df, client)
            except ImportError:
                print("openai package not installed (pip install openai>=1.0.0) — skipping LAVE.")

    results = {"accuracy": acc, "bertscore": bsc}
    if lave:
        results["lave"] = lave

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty-print matching paper Table 4 format
    has_lave = lave is not None
    header = f"{'Category':<14} {'Acc':>6} {'BScore':>8}" + (" {'LAVE':>7}" if has_lave else "")
    print(f"\n{header}")
    print("-" * (40 if has_lave else 32))
    for cat in CATEGORIES + ["overall"]:
        row = f"{cat:<14} {acc[cat]:>6.2f} {bsc[cat]:>8.2f}"
        if has_lave:
            row += f" {lave[cat]:>7.2f}"
        print(row)

    print(f"\nFull results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
