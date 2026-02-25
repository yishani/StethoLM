# StethoLM: Audio Language Model for Cardiopulmonary Analysis Across Clinical Tasks

StethoLM is an audio-language model specialized for cardiopulmonary auscultation. It connects a domain-adapted audio encoder (COLA) to a medical language model backbone (MedGemma-4B-IT) via an MLP prefix projector with LoRA fine-tuning, enabling instruction-driven clinical reasoning across seven task categories: binary classification, detection, reporting, reasoning, differential diagnosis, comparison, and location-based analysis.

**Paper:** [OpenReview](https://openreview.net/forum?id=i9RuUH9Jyj) *(TMLR)*

## Resources

| Resource | Link |
|----------|------|
| Dataset (StethoBench) | [askyishan/StethoBench](https://huggingface.co/datasets/askyishan/StethoBench) |
| Model checkpoint | [askyishan/StethoLM](https://huggingface.co/askyishan/StethoLM) |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Supervised Fine-Tuning (SFT)

```bash
python train.py \
    --input_jsonl stethobench.jsonl \
    --audio_dir /path/to/audio_files \
    --out_dir /path/to/checkpoints \
    --lm_name google/medgemma-4b-it \
    --audio_encoder_type cola \
    --epochs 30 \
    --batch_size 16 \
    --lr 5e-5 \
    --warmup_steps 100
```

### 2. Inference

```bash
python predict.py \
    --input_jsonl stethobench.jsonl \
    --output_jsonl predictions.jsonl \
    --audio_dir /path/to/audio_files \
    --checkpoint /path/to/best_gemma.pt \
    --model_name google/medgemma-4b-it \
    --audio_encoder cola \
    --split test \
    --batch_size 8 \
    --num_beams 5 \
    --max_gen_length 240
```

### 3. Multimodal DPO (mDPO)

mDPO refines the SFT model using preference optimization. It runs in two steps:

**Step 1 — Build preference pairs** (samples K=5 candidates per training example at varying temperatures, ranks by BERTScore):

```bash
python build_dpo_pairs.py \
    --input_jsonl stethobench.jsonl \
    --output_jsonl dpo_pairs.jsonl \
    --audio_dir /path/to/audio_files \
    --checkpoint /path/to/best_gemma.pt \
    --model_name google/medgemma-4b-it \
    --audio_encoder cola \
    --n_pairs 2400 \
    --n_candidates 5 \
    --temperatures 0.7 0.85 1.0 1.15 1.3
```

**Step 2 — mDPO training:**

```bash
python dpo_mdpo.py \
    --audio_dir /path/to/audio_files \
    --data_path dpo_pairs.jsonl \
    --lm_checkpoint /path/to/best_gemma.pt \
    --out_dir /path/to/mdpo_checkpoints \
    --lm_name google/medgemma-4b-it \
    --audio_encoder_type cola \
    --epochs 3 \
    --beta 0.5 \
    --lambda_conditional 0.5 \
    --lambda_anchor 1.5 \
    --lr 3e-6
```

## Citation

```bibtex
@article{stetholm2025,
  title     = {StethoLM: Audio Language Model for Cardiopulmonary Analysis Across Clinical Tasks},
  author    = {},
  journal   = {Transactions on Machine Learning Research},
  year      = {2025},
  url       = {https://openreview.net/forum?id=i9RuUH9Jyj}
}
```
