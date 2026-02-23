import torch
import torch.nn as nn
import torchaudio
from opera.src.model.models_cola import Cola
from transformers import ASTModel, ASTFeatureExtractor
from msclap import CLAP

def initialize_pretrained_model(pretrain=None):
    if pretrain == "clap":
        return CLAP(version='2023', use_cuda=False)
    elif pretrain == "ast":
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", trust_remote_code=True)
        feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        return model, feature_extractor
    elif pretrain == "cola":
        return Cola(encoder="efficientnet")
