import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import augly.audio as audaugs
from transformers import AutoTokenizer


class ALMDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        data_path: str,
        prefix_length: int,
        model_type: str,
        target_audio_seconds: int,
        max_seq_length: int,
        mode: str,
        audio_model_type: str,  # "cola", "clap", "ast"
        apply_augmentation: bool = True
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.data_path = data_path
        self.prefix_len = prefix_length
        self.model_type = model_type
        self.target_audio_seconds = target_audio_seconds
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.audio_model_type = audio_model_type
        self.apply_augmentation = apply_augmentation and (mode == "train")  # Only augment during training
        
        self.sample_rate = 16000
        self.audio_root = audio_dir

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)       
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load data
        self.data = pd.read_json(data_path, lines=True) 
        if mode == "train" or mode == "dpo":
            self.data = self.data[self.data['split'] == 'train']
        elif mode == "val":
            self.data = self.data[self.data['split'] == 'val']
        elif mode == 'test':
            self.data = self.data[self.data['split'] == 'test']
        elif mode == 'inference':
            self.data = self.data[self.data['split'] == 'inference']
    
    def __len__(self):
        return len(self.data)
    
    def _read_audio_sample(self, audio_path):
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
        return waveform
    
    def apply_random_augmentation(self, audio):
        audio = audio.astype(np.float32)
        augmentations = [
            lambda audio: audaugs.change_volume(audio, volume_db=5.0)[0],
            lambda audio: audaugs.normalize(audio)[0],
            lambda audio: audaugs.low_pass_filter(audio, cutoff_hz=300)[0],
            lambda audio: audaugs.high_pass_filter(audio, cutoff_hz=3000)[0]
        ]
        return random.choice(augmentations)(audio)
    
    def _pre_process_audio_mel_t(self, audio, n_mels=64, f_min=50, f_max=8000, nfft=1024, hop=512):
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        if audio.ndim == 2:
            audio = audio.squeeze(0)
            
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            n_fft=nfft,
            hop_length=hop
        )
        S = librosa.power_to_db(S, ref=np.max)
        mel_db = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
        target_frames = int(self.target_audio_seconds * self.sample_rate / hop)

        if mel_db.shape[1] < target_frames:
            pad_width = ((0, 0), (0, target_frames - mel_db.shape[1]))
            mel_db = np.pad(mel_db, pad_width, mode='constant')
        else:
            mel_db = mel_db[:, :target_frames]

        return torch.tensor(mel_db, dtype=torch.float32)

    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        
        audio_path = os.path.join(self.audio_root, entry["filename"])
        waveform = self._read_audio_sample(audio_path)
        target_length = int(self.target_audio_seconds * self.sample_rate)
        
        if len(waveform) > target_length:
            max_start = len(waveform) - target_length
            start = np.random.randint(0, max_start + 1) if self.mode == "train" else 0
            waveform = waveform[start:start + target_length]
        else:
            waveform = np.pad(
                waveform, 
                (0, max(0, target_length - len(waveform))),
                mode='constant'
            )
        
        raw_waveform = waveform.copy()
        
        # Apply augmentation only during training
        if self.apply_augmentation:
            waveform = self.apply_random_augmentation(waveform)
        
        spectrogram = self._pre_process_audio_mel_t(waveform)
        
        # Get domain-specific prompt
        if entry["dataset"] in ['heart-zch', 'heart-circor', 'cinc', 'BMD-HS']:
            add_prompt = 'This is a recording of heart sound.'
        elif entry["dataset"] in ['kauh']:
            add_prompt = 'This is a piece of cardiopulmonary audio.'
        elif entry["dataset"] == 'safety':
            add_prompt = ''
        else:
            add_prompt = 'This is a piece of respiratory audio.'
        
        if "medgemma" in self.model_type.lower():
            instruction = "<end_of_image>" + add_prompt + entry["instruction"]
        else:
            instruction = add_prompt + entry["instruction"]
        
        response = entry["response"]
        
        instruction_tokens = self.tokenizer(
            instruction,
            padding=False,
            add_special_tokens=False, 
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        response_tokens = self.tokenizer(
            response,
            padding=False,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Add EOS token after response
        eos_token = torch.tensor([self.tokenizer.eos_token_id])
        
        # Combine instruction + response + EOS
        combined_tokens = torch.cat([
            instruction_tokens['input_ids'].squeeze(0),
            response_tokens['input_ids'].squeeze(0),
            eos_token
        ])

        attention_mask = torch.ones_like(combined_tokens)
        input_length = len(instruction_tokens['input_ids'].squeeze(0))
        
        if self.audio_model_type == "clap":
            audio_file_paths = [audio_path]
            return audio_file_paths, combined_tokens, attention_mask, input_length
        elif self.audio_model_type == "ast":
            return raw_waveform, combined_tokens, attention_mask, input_length
        else:
            return spectrogram, combined_tokens, attention_mask, input_length
    
    def collate_fn(self, batch):
        prefix, tokens_list, attention_masks, input_lengths = zip(*batch)
        
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0
        )

        input_lengths = torch.tensor(input_lengths)
        
        if self.audio_model_type == "cola":
            spectrograms = torch.stack(prefix)
            return spectrograms, padded_tokens, padded_masks, input_lengths
        elif self.audio_model_type == "clap":
            audio_file_paths = list(prefix)
            return audio_file_paths, padded_tokens, padded_masks, input_lengths
        elif self.audio_model_type == "ast":
            raw_waveforms = list(prefix)
            return raw_waveforms, padded_tokens, padded_masks, input_lengths


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn 
    )
    
    
    
    
    



