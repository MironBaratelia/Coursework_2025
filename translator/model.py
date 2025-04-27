import torch
from transformers import MarianMTModel, MarianConfig, PreTrainedTokenizerFast

def initialize_model(
    tokenizer: PreTrainedTokenizerFast,
    dropout: float = 0.1,
    max_position_embeddings: int = 64,
    encoder_layers: int = 8,
    decoder_layers: int = 8,
    encoder_ffn_dim: int = 2048,
    decoder_ffn_dim: int = 2048,
    encoder_attention_heads: int = 12,
    decoder_attention_heads: int = 12,
    d_model: int = 768,
    activation_function: str = "gelu",
    scale_embedding: bool = True,
    add_final_layer_norm: bool = True,
    share_encoder_decoder_embeddings: bool = False,
) -> MarianMTModel:

    config = MarianConfig(
        vocab_size=tokenizer.vocab_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        d_model=d_model,
        dropout=dropout,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        scale_embedding=scale_embedding,
        activation_function=activation_function,
        add_final_layer_norm=add_final_layer_norm,
        share_encoder_decoder_embeddings=share_encoder_decoder_embeddings,
    )
    model = MarianMTModel(config)

    for param in model.get_encoder().embed_tokens.parameters():
        param.requires_grad = True
    for param in model.get_decoder().embed_tokens.parameters():
        param.requires_grad = True

    return model 