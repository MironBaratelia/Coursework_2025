import torch
from transformers import MarianMTModel, MarianConfig
import wandb

def initialize_model(tokenizer):
    config = MarianConfig(
        vocab_size=tokenizer.vocab_size,
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        d_model=768,
        dropout=wandb.config.dropout,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        label_smoothing=0.1,
        decoder_start_token_id=tokenizer.bos_token_id,
        scale_embedding=True,
        activation_function="swish",
        add_final_layer_norm=True,
        share_encoder_decoder_embeddings=False,
    )
    model = MarianMTModel(config)
    
    for param in model.get_encoder().embed_tokens.parameters():
        param.requires_grad = True
    for param in model.get_decoder().embed_tokens.parameters():
        param.requires_grad = True
    
    return model 