import os
import json
import torch
import argparse
from tqdm import tqdm
from dataloader import ALMDataset, create_dataloader
from alm import ALMModel

torch.set_float32_matmul_precision('high')


def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load checkpoint with support for both old and new formats.
    
    Old format: Just model weights
    New format: Dict with 'model_state_dict', 'optimizer_state_dict', etc.
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's new format (dict with keys) or old format (just state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("✓ Detected new checkpoint format")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Print checkpoint info if available
            if 'epoch' in checkpoint:
                print(f"✓ Checkpoint from epoch: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print("✓ Detected old checkpoint format")
            model.load_state_dict(checkpoint, strict=False)
        
        print(f"✓ Model loaded successfully")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        raise


def generate_predictions(args):
    """Main generation function."""
    
    print("\n" + "="*60)
    print("PREDICTION CONFIGURATION")
    print("="*60)
    print(f"Input file:      {args.input_jsonl}")
    print(f"Output file:     {args.output_jsonl}")
    print(f"Model:           {args.model_name}")
    print(f"Audio encoder:   {args.audio_encoder}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Max gen length:  {args.max_gen_length}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-p:           {args.top_p}")
    print(f"Num beams:       {args.num_beams}")
    if args.max_samples:
        print(f"Max samples:     {args.max_samples}")
    print("="*60 + "\n")

    # --- Load model ---
    print("Initializing model...")
    model = ALMModel(
        audio_encoder_type=args.audio_encoder,
        audio_checkpoint_path=args.audio_checkpoint,
        lm_name=args.model_name,
        device=args.device,
        prefix_size=args.prefix_size,
        prefix_length=args.prefix_length,
        max_input_length=args.max_input_length,
    )
    
    # Load trained weights
    load_model_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    # --- Load dataset ---
    print("Loading dataset...")
    dataset = ALMDataset(
        audio_dir=args.audio_dir,
        data_path=args.input_jsonl,
        model_type=args.model_name,
        prefix_length=args.prefix_length,
        target_audio_seconds=args.audio_len,
        max_seq_length=args.max_input_length,
        mode=args.split,
        audio_model_type=args.audio_encoder
    )
    
    total_samples = len(dataset)
    if args.max_samples:
        total_samples = min(total_samples, args.max_samples)
    
    print(f"✓ Loaded {len(dataset)} samples")
    print(f"✓ Will process {total_samples} samples\n")
    
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    sample_idx = 0
    skipped_batches = 0

    # --- Generate ---
    print("Generating predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batches")):
            
            # Check if we've hit max_samples limit
            if args.max_samples and sample_idx >= args.max_samples:
                print(f"\n✓ Reached max_samples limit ({args.max_samples})")
                break
            
            spectrograms, tokens, attention_masks, input_lengths = batch
            batch_size_actual = tokens.size(0) if not isinstance(tokens, list) else len(tokens)

            try:
                # Generate with configurable parameters
                gen_texts = model.generate(
                    spectrograms=spectrograms,
                    tokens=tokens,
                    attention_masks=attention_masks,
                    input_lengths=input_lengths,
                    max_new_tokens=args.max_gen_length,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0,
                )

                # Process each sample in batch
                for i, gen_text in enumerate(gen_texts):
                    if args.max_samples and sample_idx >= args.max_samples:
                        break
                    
                    row = dataset.data.iloc[sample_idx]
                    
                    result = {
                        "filename": row["filename"],
                        "instruction": row["instruction"],
                        "response": row["response"],
                        "generated": gen_text,
                    }
                    
                    # Add optional fields if they exist
                    optional_fields = ["task", "distribution", "dataset", "split", "model"]
                    for field in optional_fields:
                        if field in row:
                            result[field] = row[field]
                    
                    results.append(result)
                    sample_idx += 1
                    
                    # Print sample output every N samples
                    if args.verbose and sample_idx % args.print_every == 0:
                        print(f"\n{'─'*60}")
                        print(f"Sample {sample_idx}:")
                        print(f"Instruction: {row['instruction'][:100]}...")
                        print(f"Ground truth: {row['response'][:100]}...")
                        print(f"Generated: {gen_text[:100]}...")
                        print(f"{'─'*60}\n")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠ OOM at batch {batch_idx} (sample {sample_idx}), skipping...")
                    skipped_batches += 1
                    torch.cuda.empty_cache()
                    
                    # Skip this batch's samples
                    sample_idx += batch_size_actual
                    continue
                else:
                    print(f"\n✗ Error at batch {batch_idx}: {e}")
                    raise e

    # --- Save results ---
    print(f"\n{'='*60}")
    print("Saving results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(results)} predictions to {args.output_jsonl}")
    if skipped_batches > 0:
        print(f"⚠ Skipped {skipped_batches} batches due to OOM")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate predictions with trained ALM model")
    
    # Input/Output
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio directory")
    
    # Model paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint")
    parser.add_argument("--audio_checkpoint", type=str, default=None, help="Audio encoder base checkpoint (optional — not needed when adapter already contains fine-tuned encoder weights)")
    parser.add_argument("--model_name", type=str, default="google/medgemma-4b-it", help="LM name")
    parser.add_argument("--audio_encoder", type=str, default="cola", choices=["cola", "ast", "clap"])
    
    # Model config
    parser.add_argument("--prefix_size", type=int, default=1280)
    parser.add_argument("--prefix_length", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--audio_len", type=int, default=7, help="Audio length in seconds")
    
    # Generation config
    parser.add_argument("--max_gen_length", type=int, default=240, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    
    # Data config
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (None=all)")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Print sample outputs")
    parser.add_argument("--print_every", type=int, default=50, help="Print every N samples")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")
    if args.audio_checkpoint is not None and not os.path.exists(args.audio_checkpoint):
        raise FileNotFoundError(f"Audio checkpoint not found: {args.audio_checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    
    # Generate predictions
    results = generate_predictions(args)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()