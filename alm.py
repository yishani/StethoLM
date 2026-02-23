import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from mapper import MLP
from audio_encoder import initialize_pretrained_model
from peft import LoraConfig, get_peft_model


class ALMModel(nn.Module):
    def __init__(
        self, 
        audio_encoder_type: str,   # "cola", "ast", or "clap"
        lm_name: str,
        prefix_size: int,
        prefix_length: int,
        max_input_length: int,
        device: str,
        audio_checkpoint_path: str = None,
        freeze_audio_encoder: bool = False,
        freeze_lm_backbone: bool = False,
        freeze_mapper: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_checkpoint_path = audio_checkpoint_path
        self.lm_name = lm_name
        self.prefix_size = prefix_size
        self.prefix_length = prefix_length
        self.max_input_length = max_input_length
        self.device = device
        self.audio_encoder_type = audio_encoder_type.lower()

        # Load audio encoder
        self._load_audio_encoder(freeze_audio_encoder)
        
        # Load tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=False)
        self.tokenizer.padding_side = "right"
        self.pad_token = self.tokenizer.eos_token 
        self.pad_token_id = self.tokenizer.eos_token_id
        
        # Load language model
        self._load_language_model(freeze_lm_backbone, lora_r, lora_alpha, lora_dropout)
        
        # Initialize prefix projector
        self._init_prefix_projector()
        
        # Freeze mapper if requested
        if freeze_mapper:
            for param in self.prefix_project.parameters():
                param.requires_grad = False
            self.prefix_project.eval()
            print("✓ Prefix mapper frozen")
    
    def _load_audio_encoder(self, freeze: bool):
        """Load and configure audio encoder."""
        if self.audio_encoder_type == "cola":
            self.audio_encoder = initialize_pretrained_model(self.audio_encoder_type)
            if self.audio_checkpoint_path is not None:
                checkpoint = torch.load(self.audio_checkpoint_path, map_location='cpu')
                self.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'], strict=False)
            self.audio_encoder = self.audio_encoder.encoder.to(self.device, dtype=torch.float32)
        
        elif self.audio_encoder_type == "clap":
            self.audio_encoder = initialize_pretrained_model(self.audio_encoder_type)
            self.prefix_size = 1024  # CLAP embedding dimension
        
        elif self.audio_encoder_type == "ast":
            self.audio_encoder, self.ast_feature_extractor = initialize_pretrained_model(self.audio_encoder_type)
            self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.float32)
            self.prefix_size = 768  # AST embedding dimension
        else:
            raise ValueError(f"Unknown audio encoder type: {self.audio_encoder_type}")
        
        # Freeze if requested
        if freeze:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print(f"✓ Audio encoder ({self.audio_encoder_type}) frozen")
        else:
            print(f"✓ Audio encoder ({self.audio_encoder_type}) trainable")
    
    def _load_language_model(self, freeze_backbone: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
        """Load language model with LoRA."""
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        is_medgemma = "medgemma" in self.lm_name.lower()

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.lm_name, 
            torch_dtype=model_dtype
        )

        # Configure LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        if is_medgemma:
            # MedGemma specific handling
            self.medgemma_full = get_peft_model(base_model, peft_config).to(self.device).to(dtype=torch.float32)
            base_model_unwrapped = self.medgemma_full.base_model.model
            self.lm = base_model_unwrapped.language_model.to(dtype=torch.float32)
            self.lm_head = base_model_unwrapped.lm_head
            # self.lm_head = base_model_unwrapped.get_output_embeddings()

        else:
            # Standard LLMs
            base_model = base_model.to(self.device)
            self.lm = get_peft_model(base_model, peft_config).to(dtype=torch.float32)
            self.lm_head = None  # Use model's built-in lm_head

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.lm.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False
            print("✓ LM backbone frozen (LoRA only)")
        else:
            print("✓ LM trainable with LoRA")

        self.lm_embedding_size = self.lm.get_input_embeddings().embedding_dim
    
    def _init_prefix_projector(self):
        """Initialize MLP to project audio features to LM embedding space."""
        self.prefix_project = MLP((
            self.prefix_size,
            (self.lm_embedding_size * self.prefix_length) // 2,
            self.lm_embedding_size * self.prefix_length,
            self.lm_embedding_size * self.prefix_length
        )).to(self.device)
        print(f"✓ Prefix projector initialized: {self.prefix_size} -> {self.lm_embedding_size * self.prefix_length}")

    def encode_audio(self, audio_input):
        """
        Encode audio input to features.
        
        Args:
            audio_input: Can be spectrograms (tensor), file paths (list), or waveforms (list)
        
        Returns:
            audio_features: (batch_size, prefix_size)
        """
        if self.audio_encoder_type == "cola":
            # Input: spectrograms (batch_size, n_mels, time)
            audio_input = audio_input.to(self.device, non_blocking=True).to(torch.float32)
            audio_features = self.audio_encoder(audio_input).to(torch.float32)
        
        elif self.audio_encoder_type == "clap":
            # Input: list of file paths
            audio_file_paths = [p for sublist in audio_input for p in sublist] if isinstance(audio_input[0], list) else audio_input
            audio_embeddings = self.audio_encoder.get_audio_embeddings(audio_file_paths)
            audio_features = torch.tensor(audio_embeddings, dtype=torch.float32, device=self.device)
        
        elif self.audio_encoder_type == "ast":
            # Input: list of waveforms
            if isinstance(audio_input, list):
                inputs = self.ast_feature_extractor(
                    audio_input, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )
                input_values = inputs['input_values'].to(self.device)
                audio_outputs = self.audio_encoder(input_values)
                audio_features = audio_outputs.pooler_output
            else:
                raise ValueError("AST encoder expects list of waveforms")
        
        return audio_features

    def forward(self, audio_input, tokens, attention_masks, input_lengths, output_attentions=False):
        """
        Forward pass.
        
        Args:
            audio_input: Audio data (format depends on encoder type)
            tokens: (batch_size, seq_len) - Combined instruction + response + EOS
            attention_masks: (batch_size, seq_len)
            input_lengths: (batch_size,) - Length of instruction portion
            output_attentions: Whether to return attention weights
        
        Returns:
            logits: (batch_size, prefix_length + seq_len, vocab_size)
            attentions: (optional) Attention weights
        """
        tokens = tokens.to(self.device, non_blocking=True)
        attention_masks = attention_masks.to(self.device, non_blocking=True)
        input_lengths = input_lengths.to(self.device, non_blocking=True)
        
        batch_size = tokens.size(0)
        
        # Encode audio
        audio_features = self.encode_audio(audio_input)
        
        # Project to LM embedding space
        prefix_projections = self.prefix_project(audio_features).to(torch.float32)
        prefix_projections = prefix_projections.view(
            batch_size, self.prefix_length, self.lm_embedding_size
        )
    
        # Create placeholder tokens for audio prefix
        prefix_placeholder_ids = torch.full(
            (batch_size, self.prefix_length),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        # Concatenate prefix placeholder + text tokens
        input_ids = torch.cat([prefix_placeholder_ids, tokens], dim=1)
        attention_masks = torch.cat([
            torch.ones(batch_size, self.prefix_length, device=self.device, dtype=torch.float32),
            attention_masks.to(torch.float32)
        ], dim=1)
        
        # Get text embeddings and replace prefix with audio projections
        embedding = self.lm.get_input_embeddings()(input_ids).to(torch.float32)
        embedding[:, :self.prefix_length, :] = prefix_projections

        # Forward through LM
        output = self.lm(
            inputs_embeds=embedding, 
            attention_mask=attention_masks, 
            output_attentions=output_attentions
        )
        
        # Get logits
        if "medgemma" in self.lm_name.lower() and self.lm_head is not None:
            hidden_states = output.last_hidden_state
            logits = self.lm_head(hidden_states)
        else:
            logits = output.logits
        
        if output_attentions:
            return logits, output.attentions
        else:
            return logits

    def generate(self, spectrograms, tokens, attention_masks, input_lengths, max_new_tokens=128, **generation_kwargs):
        with torch.no_grad():
            
            batch_size = tokens.size(0) if tokens is not None else (len(spectrograms) if isinstance(spectrograms, list) else spectrograms.size(0))
            
            # Initialize to avoid scope issues
            prompt_tokens = None
            prompt_attention = None

            # --- Prepare audio features if available ---
            if spectrograms is not None:
                audio_features = self.encode_audio(spectrograms)
                prefix_projections = self.prefix_project(audio_features).to(torch.float32)
                prefix_projections = prefix_projections.view(batch_size, self.prefix_length, self.lm_embedding_size)
            else:
                prefix_projections = None

            # --- Prepare text input if available ---
            if tokens is not None and attention_masks is not None and input_lengths is not None:
                tokens = tokens.to(self.device, non_blocking=True)
                attention_masks = attention_masks.to(self.device, non_blocking=True)
                input_lengths = input_lengths.to(self.device, non_blocking=True)
                
                # Handles variable length with broadcasting
                # Slice per sample then pad to max length in batch
                seqs = [tokens[i, :input_lengths[i]] for i in range(batch_size)]
                attn_seqs = [attention_masks[i, :input_lengths[i]] for i in range(batch_size)]
                prompt_tokens = torch.nn.utils.rnn.pad_sequence(
                    seqs, batch_first=True, padding_value=self.pad_token_id
                )
                prompt_attention = torch.nn.utils.rnn.pad_sequence(
                    attn_seqs, batch_first=True, padding_value=0
                )

            # --- Compose input tokens and embeddings ---
            if prefix_projections is not None and prompt_tokens is not None:
                # Case 1: Audio + Text
                prefix_tokens = torch.full((batch_size, self.prefix_length), self.pad_token_id, dtype=torch.long, device=self.device)
                input_tokens = torch.cat([prefix_tokens, prompt_tokens], dim=1)
                attention_masks = torch.cat([
                    torch.ones(batch_size, self.prefix_length, device=self.device, dtype=torch.float32),
                    prompt_attention.to(torch.float32)
                ], dim=1)
                
                input_embeddings = self.lm.get_input_embeddings()(input_tokens).to(torch.float32)
                
                input_embeddings[:, :self.prefix_length, :] = prefix_projections

            elif prefix_projections is not None:
                # Case 2: Audio-only
                
                # --- Step 1: define hint prompt ---
                prompt_text = "This is a piece of cardiopulmonary audio."
                prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                prompt_len = prompt_tokens.shape[1]
                
                # Repeat the prompt across batch
                prompt_tokens = prompt_tokens.repeat(batch_size, 1)  # shape: (batch_size, prompt_len)
                prompt_embeddings = self.lm.get_input_embeddings()(prompt_tokens).float()  # (B, prompt_len, emb_dim)
                
                # --- Step 2: get audio prefix embeddings ---
                input_tokens = torch.full((batch_size, self.prefix_length), self.pad_token_id, device=self.device, dtype=torch.long)
                audio_embeddings = self.lm.get_input_embeddings()(input_tokens).float()
                audio_embeddings[:, :, :] = prefix_projections  # (B, prefix_len, emb_dim)
                
                # --- Step 3: concatenate prompt + audio ---
                input_embeddings = torch.cat([audio_embeddings, prompt_embeddings], dim=1)  # (B, prompt_len + prefix_len, emb_dim)
                attention_masks = torch.ones(batch_size, self.prefix_length + prompt_len, device=self.device, dtype=torch.float32)

            elif prompt_tokens is not None:
                # Case 3: Text-only
                input_tokens = prompt_tokens
                attention_masks = prompt_attention.to(torch.float32)
                input_embeddings = self.lm.get_input_embeddings()(input_tokens).to(torch.float32)
            
            if "medgemma" in self.lm_name.lower():
                gen_params = dict(
                    min_new_tokens=1,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    length_penalty=0.5,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_params.update(generation_kwargs)
                outputs = self.medgemma_full.generate(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_masks,
                    **gen_params
                )

            else:
                outputs = self.lm.generate(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_masks,
                    min_new_tokens=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )  
            
            # --- Decode everything at once ---
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return generated_texts