"""
Build preference pairs for mDPO training.

For each sampled training example:
  1. Generate K=5 candidate responses at temperatures T in [0.7, 0.85, 1.0, 1.15, 1.3]
  2. Score all K candidates with BERTScore against ground truth
  3. Select best (highest BERTScore) as response_chosen, worst as response_rejected

Output JSONL contains all original columns plus:
  - response_chosen, response_rejected, bertscore_chosen, bertscore_rejected
"""

import os
import json
import random
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from bert_score import score as bert_score_fn

from dataloader import ALMDataset
from alm import ALMModel


def load_model(args):
    model = ALMModel(
        audio_encoder_type=args.audio_encoder,
        audio_checkpoint_path=args.audio_checkpoint,
        lm_name=args.model_name,
        device=args.device,
        prefix_size=args.prefix_size,
        prefix_length=args.prefix_length,
        max_input_length=args.max_input_length,
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Build mDPO preference pairs via BERTScore ranking")

    # I/O
    parser.add_argument("--input_jsonl", type=str, required=True, help="StethoBench JSONL (train split used)")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output preference pairs JSONL")
    parser.add_argument("--audio_dir", type=str, required=True)

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="SFT model checkpoint")
    parser.add_argument("--audio_checkpoint", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--audio_encoder", type=str, default="cola", choices=["cola", "ast", "clap"])
    parser.add_argument("--prefix_size", type=int, default=1280)
    parser.add_argument("--prefix_length", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--audio_len", type=int, default=7)

    # Pair building
    parser.add_argument("--n_pairs", type=int, default=2400, help="Number of training examples to sample")
    parser.add_argument("--n_candidates", type=int, default=5, help="Candidate responses per example (K)")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.7, 0.85, 1.0, 1.15, 1.3],
                        help="Sampling temperatures, one per candidate")
    parser.add_argument("--max_gen_length", type=int, default=240)

    # Runtime
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    assert len(args.temperatures) == args.n_candidates, (
        f"len(temperatures)={len(args.temperatures)} must equal n_candidates={args.n_candidates}"
    )

    # --- Sample training examples ---
    random.seed(args.seed)
    df = pd.read_json(args.input_jsonl, lines=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    n = min(args.n_pairs, len(train_df))
    sampled_indices = random.sample(range(len(train_df)), n)
    sampled_df = train_df.iloc[sampled_indices].reset_index(drop=True)
    print(f"Sampled {n} / {len(train_df)} training examples")

    # Save subset to temp file for ALMDataset
    tmp_path = "/tmp/dpo_build_sampled.jsonl"
    sampled_df.to_json(tmp_path, orient='records', lines=True)

    # --- Create dataset (audio loading only; no augmentation) ---
    dataset = ALMDataset(
        audio_dir=args.audio_dir,
        data_path=tmp_path,
        model_type=args.model_name,
        prefix_length=args.prefix_length,
        target_audio_seconds=args.audio_len,
        max_seq_length=args.max_input_length,
        mode="dpo",               # filters to train split, no random crop
        audio_model_type=args.audio_encoder,
        apply_augmentation=False,
    )
    print(f"Dataset: {len(dataset)} examples")

    ground_truths = [dataset.data.iloc[i]["response"] for i in range(len(dataset))]

    # --- Load model ---
    print("Loading model...")
    model = load_model(args)
    model.to(args.device)

    # --- Generate K candidates per example ---
    # Outer loop over temperatures so we can batch examples efficiently
    K = args.n_candidates
    all_candidates = [[] for _ in range(len(dataset))]

    for T in args.temperatures:
        print(f"\nGenerating candidates at temperature T={T:.2f}...")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        sample_idx = 0
        for batch in tqdm(loader, desc=f"T={T:.2f}"):
            spectrograms, tokens, masks, input_lengths = batch
            spectrograms = spectrograms.to(args.device).float()
            tokens = tokens.to(args.device)
            masks = masks.to(args.device)
            input_lengths = input_lengths.to(args.device)

            with torch.no_grad():
                gen_texts = model.generate(
                    spectrograms=spectrograms,
                    tokens=tokens,
                    attention_masks=masks,
                    input_lengths=input_lengths,
                    max_new_tokens=args.max_gen_length,
                    temperature=T,
                    do_sample=True,
                    num_beams=1,
                )

            for gen in gen_texts:
                all_candidates[sample_idx].append(gen)
                sample_idx += 1

    # --- Score all candidates with BERTScore ---
    print(f"\nScoring {len(dataset) * K} candidate-reference pairs with BERTScore...")

    # Flatten to (n_examples * K,) for batch scoring
    hypotheses = [cand for candidates in all_candidates for cand in candidates]
    references = [gt for gt in ground_truths for _ in range(K)]

    _, _, F1 = bert_score_fn(
        hypotheses,
        references,
        lang="en",
        device=args.device,
        verbose=True,
        batch_size=64,
    )
    F1 = F1.reshape(len(dataset), K)

    # --- Select best and worst per example ---
    results = []
    skipped = 0
    for i in range(len(dataset)):
        scores = F1[i]
        best_idx = scores.argmax().item()
        worst_idx = scores.argmin().item()

        if best_idx == worst_idx:
            skipped += 1
            continue  # all K candidates scored identically — skip

        row = dataset.data.iloc[i].to_dict()
        row["response_chosen"] = all_candidates[i][best_idx]
        row["response_rejected"] = all_candidates[i][worst_idx]
        row["bertscore_chosen"] = round(scores[best_idx].item(), 4)
        row["bertscore_rejected"] = round(scores[worst_idx].item(), 4)
        results.append(row)

    # --- Save ---
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    chosen_scores = [r["bertscore_chosen"] for r in results]
    rejected_scores = [r["bertscore_rejected"] for r in results]
    print(f"\nSaved {len(results)} preference pairs to {args.output_jsonl}")
    if skipped:
        print(f"Skipped {skipped} examples (all K candidates had identical BERTScore)")
    print(f"BERTScore chosen:   mean={sum(chosen_scores)/len(chosen_scores):.3f}")
    print(f"BERTScore rejected: mean={sum(rejected_scores)/len(rejected_scores):.3f}")


if __name__ == "__main__":
    main()
