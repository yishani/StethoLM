import gc
import wandb
import os
import torch
import torch.nn.functional as nnf
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from dataloader import ALMDataset, create_dataloader
from alm import ALMModel
import argparse



# ------------------ Helper Functions ------------------

def get_checkpoint_name(lm_name: str) -> str:
    """Generate checkpoint name based on language model."""
    lm_name = lm_name.lower()
    if "gemma" in lm_name:
        return "best_gemma"
    elif "llama" in lm_name:
        return "best_llama"
    else:
        return "best_model"

def save_checkpoint_locally(model_state_dict, out_dir, checkpoint_name):
    """Save model checkpoint locally."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    local_path = os.path.join(out_dir, f"{checkpoint_name}.pt")
    torch.save(model_state_dict, local_path)
    print(f">>> Checkpoint saved locally at {local_path}")
    return local_path

def save_checkpoint_to_wandb(local_path, checkpoint_name):
    """Upload checkpoint to WandB."""
    try:
        artifact = wandb.Artifact(name=checkpoint_name, type="model")
        artifact.add_file(local_path)
        wandb.log_artifact(artifact, aliases=["latest"])
        print(f">>> Checkpoint '{checkpoint_name}' uploaded/updated on WandB")
    except Exception as e:
        print(f">>> Warning: Failed to save checkpoint '{checkpoint_name}' to WandB: {e}")

# ------------------ Evaluation ------------------

def evaluate(model, val_loader, accelerator):
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, (spectrograms, tokens, attention_masks, input_lengths) in enumerate(val_loader):
            try:
                if not isinstance(spectrograms, list):
                        spectrograms = spectrograms.to(accelerator.device).float()
                tokens = tokens.to(accelerator.device).long()
                attention_masks = attention_masks.to(accelerator.device).long()
                input_lengths = input_lengths.to(accelerator.device).long()

                logits = model(spectrograms, tokens, attention_masks, input_lengths)
                batch_size = tokens.size(0)
                loss = 0.0

                for b in range(batch_size):
                    condensed_tokens = tokens[b, input_lengths[b]:]
                    valid_mask = attention_masks[b, input_lengths[b]:]
                    condensed_tokens = condensed_tokens[valid_mask == 1]

                    condensed_logits = logits[b, model.prefix_length + input_lengths[b] - 1:]
                    condensed_logits = condensed_logits[:len(condensed_tokens)]

                    loss += nnf.cross_entropy(condensed_logits, condensed_tokens)

                loss = loss / batch_size
                total_loss += loss.item()
                valid_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f">>> OOM error in validation batch {batch_idx}, skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f">>> Runtime error in validation batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                print(f">>> Unexpected error in validation batch {batch_idx}: {e}")
                continue

    if valid_batches == 0:
        print(">>> Warning: No valid batches in validation")
        return float('inf')

    return total_loss / valid_batches

# ------------------ Training ------------------

def train_model(train_loader, val_loader, model_obj, args):
    accelerator = Accelerator() # mixed_precision='no'
    device = accelerator.device
    model = model_obj.to(device)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    elif args.load_pretrained:
        checkpoint = os.path.join(args.out_dir, get_checkpoint_name(args.lm_name) + ".pt")
        if args.verbose:
            print(f">> Attempting to load pre-trained model from {checkpoint}", flush=True)
        if os.path.exists(checkpoint):
            try:
                model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
                print(">>> Pre-trained model loaded successfully from local file")
            except Exception as e:
                print(f">>> Warning: Failed to load pre-trained model from local file: {e}")
        else:
            print(f">>> No local checkpoint found at {checkpoint}. You may need to download from wandb manually.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.98))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    accelerator.wait_for_everyone()

    best_val_loss = float("inf")
    skipped_batches = 0
    best_checkpoint_path = None

    try:
        for epoch in tqdm(range(args.epochs)):
            model.train()
            total_loss = 0.0
            valid_batches = 0

            for i, (spectrograms, tokens, attention_masks, input_lengths) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Batches"
            ):
                try:
                    with accelerator.accumulate(model):
                        if not isinstance(spectrograms, list):
                            spectrograms = spectrograms.to(accelerator.device).float()
                        tokens = tokens.to(accelerator.device).long()
                        attention_masks = attention_masks.to(accelerator.device).long()
                        input_lengths = input_lengths.to(accelerator.device).long()

                        logits = model(spectrograms, tokens, attention_masks, input_lengths)
                        batch_size = tokens.size(0)
                        loss = 0.0

                        for b in range(batch_size):
                            condensed_tokens = tokens[b, input_lengths[b]:]
                            valid_mask = attention_masks[b, input_lengths[b]:]
                            condensed_tokens = condensed_tokens[valid_mask == 1]

                            condensed_logits = logits[b, model.prefix_length + input_lengths[b] - 1:]
                            condensed_logits = condensed_logits[:len(condensed_tokens)]

                            loss += nnf.cross_entropy(condensed_logits, condensed_tokens)

                        loss = loss / batch_size

                        accelerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        total_loss += loss.item()
                        valid_batches += 1
                        avg_loss = total_loss / valid_batches

                        if (i + 1) % 10 == 0:
                            wandb.log({
                                "train_loss": avg_loss,
                                "batch": i + 1,
                                "epoch": epoch + 1,
                                "skipped_batches": skipped_batches
                            })

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f">>> OOM error at epoch {epoch + 1}, batch {i + 1}, skipping batch")
                        skipped_batches += 1
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        print(f">>> Runtime error at epoch {epoch + 1}, batch {i + 1}: {e}")
                        skipped_batches += 1
                        continue

                except Exception as e:
                    print(f">>> Unexpected error at epoch {epoch + 1}, batch {i + 1}: {e}")
                    skipped_batches += 1
                    continue

            if valid_batches == 0:
                print(f">>> Warning: No valid batches in epoch {epoch + 1}")
                continue

            val_loss = evaluate(model, val_loader, accelerator)
            wandb.log({
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "epoch_skipped_batches": skipped_batches,
                "valid_batches_in_epoch": valid_batches
            })

            print(f">>> Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Skipped: {skipped_batches} batches")

            # Save best model locally only
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_name = get_checkpoint_name(args.lm_name)
                best_checkpoint_path = save_checkpoint_locally(model.state_dict(), args.out_dir, checkpoint_name)

    except KeyboardInterrupt:
        print(">>> Training interrupted by user")
    except Exception as e:
        print(f">>> Training failed with error: {e}")
        raise

    print(f">>> Training completed! Total skipped batches: {skipped_batches}")

    # Upload the best checkpoint to WandB once at the end
    # if best_checkpoint_path is not None:
    #     save_checkpoint_to_wandb(best_checkpoint_path, get_checkpoint_name(args.lm_name))

    return model

# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Train ALM Model")
    parser.add_argument("--audio_dir", type=str, default="/home/ywang3/audio_files", help="Directory containing audio files")
    parser.add_argument("--data_path", type=str, default="/home/ywang3/alm/all.jsonl", help="Path to CSV file with data")
    parser.add_argument("--audio_checkpoint", type=str, default=None, help="Path to audio encoder base checkpoint (optional)")
    parser.add_argument("--out_dir", type=str, default="/projects/prjs1728/checkpoints/opera", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--load_pretrained", action="store_true", help="Load pretrained model")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--lm_name", type=str, default=  "google/medgemma-4b-it", help="Language model name") 
    # "meta-llama/Llama-3.2-3B-Instruct" 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' 'microsoft/MediPhi-Instruct' "google/medgemma-4b-it"
    parser.add_argument("--prefix_size", type=int, default=1280, help="Audio feature dimension")
    parser.add_argument("--prefix_length", type=int, default=4, help="Number of prefix tokens")
    parser.add_argument("--audio_len", type=int, default=7, help="Audio length")
    parser.add_argument("--max_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--freeze_audio_encoder", action="store_true", help="If set, freezes the audio encoder during training")
    parser.add_argument("--freeze_lm_backbone", action="store_true", help="If set, freezes the LM backbone during training")
    parser.add_argument("--audio_encoder_type", type=str, default="cola", choices=["cola", "ast", "clap"], help="Which audio encoder to use: cola (default), ast, or clap.")

    args = parser.parse_args()

    wandb.init(project="alm-training", config=vars(args))

    print("Creating dataset...")
    train_dataset = ALMDataset(
        audio_dir=args.audio_dir,
        data_path=args.data_path,
        model_type=args.lm_name,
        prefix_length=args.prefix_length,
        target_audio_seconds=args.audio_len,
        max_seq_length=args.max_len,
        mode="train",
        audio_model_type=args.audio_encoder_type
    )

    val_dataset = ALMDataset(
        audio_dir=args.audio_dir,
        data_path=args.data_path,
        model_type=args.lm_name,
        prefix_length=args.prefix_length,
        target_audio_seconds=args.audio_len,
        max_seq_length=args.max_len,
        mode="val",
        audio_model_type=args.audio_encoder_type
    )

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing model...")
    model = ALMModel(
        audio_encoder_type=args.audio_encoder_type,
        audio_checkpoint_path=args.audio_checkpoint,
        lm_name=args.lm_name,
        device="cuda",
        prefix_size=args.prefix_size,
        prefix_length=args.prefix_length,
        max_input_length=args.max_len,
    )

    print(f"Starting training with {len(train_loader)} train batches and {len(val_loader)} val batches per epoch...")
    trained_model = train_model(train_loader, val_loader, model, args)

    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
