import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import argparse
import wandb

from dataloader import ALMDataset
from torch.utils.data import DataLoader
from alm import ALMModel

torch.set_float32_matmul_precision("high")


# ==============================
# AUDIO DEGRADATION FUNCTIONS
# ==============================
def degrade_audio_time_crop(spectrogram, crop_ratio=0.15):
    """
    Crop 10-20% of the spectrogram from random location.
    Args:
        spectrogram: (n_mels, time)
        crop_ratio: fraction to crop (0.1-0.2)
    Returns:
        degraded spectrogram with same shape (zero-padded)
    """
    n_mels, time_steps = spectrogram.shape
    crop_length = int(time_steps * crop_ratio)
    
    # Randomly choose start or end to crop
    if random.random() < 0.5:
        # Crop from start
        degraded = spectrogram[:, crop_length:]
        degraded = F.pad(degraded, (0, crop_length), value=0)
    else:
        # Crop from end
        degraded = spectrogram[:, :-crop_length]
        degraded = F.pad(degraded, (crop_length, 0), value=0)
    
    return degraded

def degrade_audio_freq_mask(spectrogram, mask_ratio=0.15):
    """
    Mask random frequency bands.
    Args:
        spectrogram: (n_mels, time)
        mask_ratio: fraction of frequency bands to mask
    Returns:
        degraded spectrogram
    """
    n_mels, time_steps = spectrogram.shape
    num_mask_bands = int(n_mels * mask_ratio)
    
    degraded = spectrogram.clone()
    mask_start = random.randint(0, n_mels - num_mask_bands)
    degraded[mask_start:mask_start + num_mask_bands, :] = 0
    
    return degraded

def degrade_audio_spectral_crop(spectrogram, crop_ratio=0.15):
    """
    Crop spectrogram in both time and frequency dimensions.
    Args:
        spectrogram: (n_mels, time)
        crop_ratio: fraction to crop
    Returns:
        degraded spectrogram with same shape
    """
    n_mels, time_steps = spectrogram.shape
    
    # Crop time dimension
    time_crop = int(time_steps * crop_ratio)
    time_start = random.randint(0, time_crop)
    time_end = time_steps - time_crop + time_start
    
    # Crop frequency dimension  
    freq_crop = int(n_mels * crop_ratio)
    freq_start = random.randint(0, freq_crop)
    freq_end = n_mels - freq_crop + freq_start
    
    # Extract cropped region and pad back to original size
    cropped = spectrogram[freq_start:freq_end, time_start:time_end]
    degraded = torch.zeros_like(spectrogram)
    degraded[freq_start:freq_end, time_start:time_end] = cropped
    
    return degraded

def degrade_audio(spectrogram, method='time_crop'):
    """
    Apply audio degradation.
    Args:
        spectrogram: (n_mels, time) or (batch, n_mels, time)
        method: 'time_crop', 'freq_mask', or 'spectral_crop'
    """
    if spectrogram.dim() == 3:
        # Batch of spectrograms
        return torch.stack([degrade_audio(s, method) for s in spectrogram])
    
    if method == 'time_crop':
        return degrade_audio_time_crop(spectrogram, crop_ratio=random.uniform(0.1, 0.2))
    elif method == 'freq_mask':
        return degrade_audio_freq_mask(spectrogram, mask_ratio=random.uniform(0.1, 0.2))
    elif method == 'spectral_crop':
        return degrade_audio_spectral_crop(spectrogram, crop_ratio=random.uniform(0.1, 0.2))
    else:
        # Random choice
        return degrade_audio(spectrogram, method=random.choice(['time_crop', 'freq_mask', 'spectral_crop']))


# ==============================
# CUSTOM DATASET FOR mDPO
# ==============================
class mDPODataset(ALMDataset):
    """
    Extended dataset that returns both original and degraded audio.
    """
    def __init__(self, *args, degradation_method='random', **kwargs):
        super().__init__(*args, **kwargs)
        self.degradation_method = degradation_method
    
    def __getitem__(self, idx):
        # Get original data (returns audio for chosen response)
        audio_chosen, tokens_chosen, mask_chosen, input_len = super().__getitem__(idx)
        
        # Get data for rejected response
        entry = self.data.iloc[idx]
        
        # Create rejected response tokens
        if "medgemma" in self.model_type.lower():
            add_prompt = self._get_add_prompt(entry["dataset"])
            instruction = "<end_of_image>" + add_prompt + entry["instruction"]
        else:
            add_prompt = self._get_add_prompt(entry["dataset"])
            instruction = add_prompt + entry["instruction"]

        response_rejected = entry["response_rejected"]
        
        instruction_tokens = self.tokenizer(
            instruction,
            padding=False,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        response_tokens_rejected = self.tokenizer(
            response_rejected,
            padding=False,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        eos_token = torch.tensor([self.tokenizer.eos_token_id])
        tokens_rejected = torch.cat([
            instruction_tokens['input_ids'].squeeze(0),
            response_tokens_rejected['input_ids'].squeeze(0),
            eos_token
        ])
        mask_rejected = torch.ones_like(tokens_rejected)
        
        # Degrade audio for conditional preference
        if self.audio_model_type == "cola":
            audio_degraded = degrade_audio(audio_chosen, method=self.degradation_method)
        else:
            # For CLAP/AST, we'll handle degradation differently
            audio_degraded = audio_chosen  # Placeholder
        
        return (audio_chosen, audio_degraded, 
                tokens_chosen, tokens_rejected, 
                mask_chosen, mask_rejected, 
                input_len)
    
    def _get_add_prompt(self, dataset):
        """Get domain-specific prompt."""
        if dataset in ['heart-zch', 'heart-circor', 'cinc', 'BMD-HS']:
            return 'This is a recording of heart sound.'
        elif dataset in ['kauh']:
            return 'This is a piece of cardiopulmonary audio.'
        elif dataset == 'safety':
            return ''
        else:
            return 'This is a piece of respiratory audio.'
    
    def collate_fn(self, batch):
        """Custom collate for mDPO data."""
        audio_chosen_list, audio_degraded_list = [], []
        tokens_chosen_list, tokens_rejected_list = [], []
        mask_chosen_list, mask_rejected_list = [], []
        input_len_list = []
        
        for item in batch:
            audio_chosen, audio_degraded, tok_c, tok_r, mask_c, mask_r, input_len = item
            audio_chosen_list.append(audio_chosen)
            audio_degraded_list.append(audio_degraded)
            tokens_chosen_list.append(tok_c)
            tokens_rejected_list.append(tok_r)
            mask_chosen_list.append(mask_c)
            mask_rejected_list.append(mask_r)
            input_len_list.append(input_len)
        
        # Pad tokens
        tokens_chosen = torch.nn.utils.rnn.pad_sequence(
            tokens_chosen_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        tokens_rejected = torch.nn.utils.rnn.pad_sequence(
            tokens_rejected_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        mask_chosen = torch.nn.utils.rnn.pad_sequence(
            mask_chosen_list, batch_first=True, padding_value=0
        )
        mask_rejected = torch.nn.utils.rnn.pad_sequence(
            mask_rejected_list, batch_first=True, padding_value=0
        )
        input_lengths = torch.tensor(input_len_list)
        
        # Stack audio
        if self.audio_model_type == "cola":
            audio_chosen = torch.stack(audio_chosen_list)
            audio_degraded = torch.stack(audio_degraded_list)
        else:
            audio_chosen = audio_chosen_list
            audio_degraded = audio_degraded_list
        
        return (audio_chosen, audio_degraded,
                tokens_chosen, tokens_rejected,
                mask_chosen, mask_rejected,
                input_lengths)


# ==============================
# mDPO LOSS FUNCTIONS (FIXED)
# ==============================
def get_log_probs(model, audio, tokens, attention_masks, input_lengths):
    """
    Get length-normalized log probabilities for response tokens.
    
    KEY FIX: Returns AVERAGE log prob per token, not sum!
    """
    logits = model(audio, tokens, attention_masks, input_lengths)
    batch_size = tokens.size(0)
    
    log_probs_list = []
    seq_lengths = []
    
    for b in range(batch_size):
        # Extract response tokens
        condensed_tokens = tokens[b, input_lengths[b]:]
        valid_mask = attention_masks[b, input_lengths[b]:]
        condensed_tokens = condensed_tokens[valid_mask == 1]
        
        # Extract corresponding logits
        condensed_logits = logits[b, model.prefix_length + input_lengths[b] - 1:]
        condensed_logits = condensed_logits[:len(condensed_tokens)]
        
        # Convert to log probabilities
        log_probs = F.log_softmax(condensed_logits, dim=-1)
        
        # Get log prob of actual tokens
        token_log_probs = torch.gather(
            log_probs, 
            1, 
            condensed_tokens.unsqueeze(1)
        ).squeeze(1)
        
        # CRITICAL FIX: Average instead of sum
        sequence_length = len(condensed_tokens)
        avg_log_prob = token_log_probs.sum() / max(sequence_length, 1)
        
        log_probs_list.append(avg_log_prob)
        seq_lengths.append(sequence_length)
    
    return torch.stack(log_probs_list), torch.tensor(seq_lengths, device=audio.device if isinstance(audio, torch.Tensor) else tokens.device)


def compute_mdpo_loss(
    # Standard DPO components (response contrast)
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    # Conditional preference components (audio contrast)
    policy_chosen_clean_logps,
    policy_chosen_degraded_logps,
    reference_chosen_clean_logps,
    reference_chosen_degraded_logps,
    # Sequence lengths for monitoring
    chosen_lengths,
    rejected_lengths,
    beta=0.1,
    anchor_margin=0.0,
    lambda_conditional=1.0,
    lambda_anchor=1.0
):
    """
    Compute mDPO loss with three components.
    Now uses length-normalized log probs for numerical stability.
    """
    # Component 1: Standard DPO (contrasting responses)
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    loss_dpo = -F.logsigmoid(beta * (policy_logratios - reference_logratios))
    
    # Component 2: Conditional preference (contrasting audio)
    policy_audio_logratios = policy_chosen_clean_logps - policy_chosen_degraded_logps
    reference_audio_logratios = reference_chosen_clean_logps - reference_chosen_degraded_logps
    loss_conditional = -F.logsigmoid(beta * (policy_audio_logratios - reference_audio_logratios))
    
    # Component 3: Anchor loss (force chosen reward > margin)
    reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps)
    loss_anchor = -F.logsigmoid(reward_chosen - anchor_margin)
    
    # Combine losses
    total_loss = loss_dpo + lambda_conditional * loss_conditional + lambda_anchor * loss_anchor
    
    # Compute metrics
    reward_accuracies = (policy_logratios > 0).float()
    audio_accuracies = (policy_audio_logratios > 0).float()
    
    metrics = {
        'loss_total': total_loss.mean().item(),
        'loss_dpo': loss_dpo.mean().item(),
        'loss_conditional': loss_conditional.mean().item(),
        'loss_anchor': loss_anchor.mean().item(),
        'accuracy_response': reward_accuracies.mean().item(),
        'accuracy_audio': audio_accuracies.mean().item(),
        'margin_response': policy_logratios.mean().item(),
        'margin_audio': policy_audio_logratios.mean().item(),
        'reward_chosen': reward_chosen.mean().item(),
        'avg_chosen_length': chosen_lengths.float().mean().item(),
        'avg_rejected_length': rejected_lengths.float().mean().item(),
        # Raw log probs for debugging
        'policy_chosen_logp': policy_chosen_logps.mean().item(),
        'policy_rejected_logp': policy_rejected_logps.mean().item(),
        'reference_chosen_logp': reference_chosen_logps.mean().item(),
        'reference_rejected_logp': reference_rejected_logps.mean().item(),
    }
    
    return total_loss.mean(), metrics


# ==============================
# TRAINING FUNCTION
# ==============================
def train_mdpo(args):
    device = args.device

    # Initialize wandb
    wandb.init(
        project="audio-lm-mdpo",
        config={
            "beta": args.beta,
            "anchor_margin": args.anchor_margin,
            "lambda_conditional": args.lambda_conditional,
            "lambda_anchor": args.lambda_anchor,
            "learning_rate": args.lr,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "model": args.lm_name,
            "audio_encoder": args.audio_encoder_type,
            "degradation_method": args.degradation,
        }
    )

    # Load data
    import pandas as pd
    dpo_data = pd.read_json(args.data_path, lines=True)
    dpo_data = dpo_data[dpo_data['split'] == 'train'].reset_index(drop=True)
    print(f"[LOADED] mDPO data: {len(dpo_data)} examples")

    # Save temporary data file for dataset
    temp_jsonl = "/tmp/mdpo_data.jsonl"
    dpo_data.to_json(temp_jsonl, orient='records', lines=True)

    # Create dataset
    dataset = mDPODataset(
        audio_dir=args.audio_dir,
        data_path=temp_jsonl,
        model_type=args.lm_name,
        prefix_length=args.prefix_length,
        target_audio_seconds=args.audio_len,
        max_seq_length=args.max_input_length,
        mode="dpo",
        audio_model_type=args.audio_encoder_type,
        degradation_method=args.degradation
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    print(f"[DATASET] {len(dataloader)} batches")

    # Load policy model
    policy_model = ALMModel(
        audio_checkpoint_path=args.audio_checkpoint,
        audio_encoder_type=args.audio_encoder_type,
        lm_name=args.lm_name,
        device=device,
        prefix_size=args.prefix_size,
        prefix_length=args.prefix_length,
        max_input_length=args.max_input_length,
    )
    checkpoint = torch.load(args.lm_checkpoint, map_location=device)
    policy_model.load_state_dict(checkpoint, strict=False)
    policy_model.to(device)
    
    # Set training mode correctly
    policy_model.train()
    policy_model.audio_encoder.eval()
    # policy_model.prefix_project.eval()
    
    # Freeze audio encoder and mapper
    for param in policy_model.audio_encoder.parameters():
        param.requires_grad = False
    # for param in policy_model.prefix_project.parameters():
    #    param.requires_grad = False
    
    print("✓ Policy model loaded")
    print(f"  Audio encoder training: {policy_model.audio_encoder.training}")
    print(f"  Prefix mapper training: {policy_model.prefix_project.training}")
    print(f"  LM training: {policy_model.lm.training}")
    
    # Load reference model
    reference_model = ALMModel(
        audio_checkpoint_path=args.audio_checkpoint,
        audio_encoder_type=args.audio_encoder_type,
        lm_name=args.lm_name,
        device=device,
        prefix_size=args.prefix_size,
        prefix_length=args.prefix_length,
        max_input_length=args.max_input_length,
    )
    reference_model.load_state_dict(checkpoint, strict=False)
    reference_model.to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    print("✓ Reference model loaded")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.lr
    )

    # Training loop
    beta = args.beta
    anchor_margin = args.anchor_margin
    lambda_conditional = args.lambda_conditional
    lambda_anchor = args.lambda_anchor
    num_epochs = args.epochs
    
    print("\n" + "="*60)
    print("STARTING mDPO TRAINING")
    print(f"Beta: {beta}, Anchor: {anchor_margin}")
    print(f"Lambda - Conditional: {lambda_conditional}, Anchor: {lambda_anchor}")
    print("="*60)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_metrics = {k: 0.0 for k in [
            'loss_dpo', 'loss_conditional', 'loss_anchor',
            'accuracy_response', 'accuracy_audio', 'margin_response',
            'margin_audio', 'reward_chosen', 'avg_chosen_length',
            'avg_rejected_length', 'policy_chosen_logp', 'policy_rejected_logp'
        ]}
        step = 0
        
        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            (audio_chosen, audio_degraded,
            tokens_chosen, tokens_rejected,
            mask_chosen, mask_rejected,
            input_lengths) = batch_data
            
            # Move to device
            audio_chosen = audio_chosen.to(device).float()
            audio_degraded = audio_degraded.to(device).float()
            tokens_chosen = tokens_chosen.to(device).long()
            tokens_rejected = tokens_rejected.to(device).long()
            mask_chosen = mask_chosen.to(device).long()
            mask_rejected = mask_rejected.to(device).long()
            input_lengths = input_lengths.to(device).long()
            
            # === Policy model forward passes ===
            # 1. Chosen response with clean audio
            policy_chosen_logps, chosen_lengths = get_log_probs(
                policy_model, audio_chosen, tokens_chosen, mask_chosen, input_lengths
            )
            
            # 2. Rejected response with clean audio
            policy_rejected_logps, rejected_lengths = get_log_probs(
                policy_model, audio_chosen, tokens_rejected, mask_rejected, input_lengths
            )
            
            # 3. Chosen response with degraded audio (for conditional preference)
            policy_chosen_degraded_logps, _ = get_log_probs(
                policy_model, audio_degraded, tokens_chosen, mask_chosen, input_lengths
            )
            
            # === Reference model forward passes ===
            with torch.no_grad():
                reference_chosen_logps, _ = get_log_probs(
                    reference_model, audio_chosen, tokens_chosen, mask_chosen, input_lengths
                )
                
                reference_rejected_logps, _ = get_log_probs(
                    reference_model, audio_chosen, tokens_rejected, mask_rejected, input_lengths
                )
                
                reference_chosen_degraded_logps, _ = get_log_probs(
                    reference_model, audio_degraded, tokens_chosen, mask_chosen, input_lengths
                )
            
            # Compute mDPO loss
            loss, metrics = compute_mdpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                policy_chosen_logps,  # clean audio for conditional
                policy_chosen_degraded_logps,  # degraded audio for conditional
                reference_chosen_logps,
                reference_chosen_degraded_logps,
                chosen_lengths,
                rejected_lengths,
                beta=beta,
                anchor_margin=anchor_margin,
                lambda_conditional=lambda_conditional,
                lambda_anchor=lambda_anchor
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            total_loss += metrics['loss_total']
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            step += 1
            
            # Log to wandb
            if step % 10 == 0:
                wandb.log({
                    "train/loss_total": metrics['loss_total'],
                    "train/loss_dpo": metrics['loss_dpo'],
                    "train/loss_conditional": metrics['loss_conditional'],
                    "train/loss_anchor": metrics['loss_anchor'],
                    "train/accuracy_response": metrics['accuracy_response'],
                    "train/accuracy_audio": metrics['accuracy_audio'],
                    "train/margin_response": metrics['margin_response'],
                    "train/margin_audio": metrics['margin_audio'],
                    "train/reward_chosen": metrics['reward_chosen'],
                    "train/avg_chosen_length": metrics['avg_chosen_length'],
                    "train/avg_rejected_length": metrics['avg_rejected_length'],
                    "train/step": step,
                    "train/epoch": epoch
                })
                
                if step % 100 == 0:
                    print(f"\n  Step {step}:")
                    print(f"    Loss: {metrics['loss_total']:.4f} "
                        f"(DPO: {metrics['loss_dpo']:.4f}, "
                        f"Cond: {metrics['loss_conditional']:.4f}, "
                        f"Anchor: {metrics['loss_anchor']:.4f})")
                    print(f"    Acc: Response={metrics['accuracy_response']:.3f}, "
                        f"Audio={metrics['accuracy_audio']:.3f}")
                    print(f"    Margins: Response={metrics['margin_response']:.3f}, "
                        f"Audio={metrics['margin_audio']:.3f}")
                    print(f"    Reward chosen: {metrics['reward_chosen']:.3f}")
        
        # Epoch summary
        avg_loss = total_loss / max(1, step)
        avg_metrics = {k: v / max(1, step) for k, v in total_metrics.items()}
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*60}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"  DPO Loss: {avg_metrics['loss_dpo']:.4f}")
        print(f"  Conditional Loss: {avg_metrics['loss_conditional']:.4f}")
        print(f"  Anchor Loss: {avg_metrics['loss_anchor']:.4f}")
        print(f"Accuracy: Response={avg_metrics['accuracy_response']:.3f}, "
            f"Audio={avg_metrics['accuracy_audio']:.3f}")
        print(f"Margins: Response={avg_metrics['margin_response']:.3f}, "
            f"Audio={avg_metrics['margin_audio']:.3f}")
        print(f"Reward Chosen: {avg_metrics['reward_chosen']:.3f}")
        print(f"Avg Sequence Lengths: Chosen={avg_metrics['avg_chosen_length']:.1f}, "
            f"Rejected={avg_metrics['avg_rejected_length']:.1f}")
        print(f"{'='*60}\n")
        
        wandb.log({
            "epoch/avg_loss": avg_loss,
            **{f"epoch/avg_{k}": v for k, v in avg_metrics.items()},
            "epoch/number": epoch + 1
        })
        
        # Save checkpoint
        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.out_dir, f"mdpo_epoch_{epoch+1}.pt")
        torch.save(policy_model.state_dict(), checkpoint_path)
        print(f"[SAVED] {checkpoint_path}\n")

        # Save best model
        current_accuracy = (avg_metrics['accuracy_response'] + avg_metrics['accuracy_audio']) / 2
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_path = os.path.join(args.out_dir, "mdpo_best.pt")
            torch.save(policy_model.state_dict(), best_path)
            print(f"[SAVED BEST] {best_path} (Combined accuracy: {best_accuracy:.3f})\n")
    
    wandb.finish()
    print("[DONE] mDPO training complete!")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StethoLM with mDPO")

    # Data paths
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--data_path", type=str, required=True, help="Path to DPO pairs JSONL file")
    parser.add_argument("--audio_checkpoint", type=str, default=None, help="Path to audio encoder base checkpoint (optional)")
    parser.add_argument("--lm_checkpoint", type=str, required=True, help="Path to SFT model checkpoint to start DPO from")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints")

    # Model config
    parser.add_argument("--lm_name", type=str, default="google/medgemma-4b-it", help="Language model name")
    parser.add_argument("--audio_encoder_type", type=str, default="cola", choices=["cola", "ast", "clap"])
    parser.add_argument("--prefix_size", type=int, default=1280, help="Audio feature dimension")
    parser.add_argument("--prefix_length", type=int, default=4, help="Number of prefix tokens")
    parser.add_argument("--max_input_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--audio_len", type=int, default=7, help="Audio length in seconds")

    # mDPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.5, help="DPO temperature")
    parser.add_argument("--anchor_margin", type=float, default=0.5, help="Anchor loss margin")
    parser.add_argument("--lambda_conditional", type=float, default=0.5, help="Weight for conditional preference loss")
    parser.add_argument("--lambda_anchor", type=float, default=1.5, help="Weight for anchor loss")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # Audio degradation
    parser.add_argument("--degradation", type=str, default="random",
                        choices=["time_crop", "freq_mask", "spectral_crop", "random"],
                        help="Audio degradation method for conditional preference")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train_mdpo(args)