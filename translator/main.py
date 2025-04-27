import os
import logging
import argparse

from train import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    first_run_config = {
        "data_dir": "tsv_data",
        "tokenizer_path": "tokenizer",
        "output_dir": "./apsua_translator",
        "logging_dir": "./logs",

        "dropout": 0.2,
        "max_position_embeddings": 64,
        "encoder_layers": 8,
        "decoder_layers": 8,
        "encoder_ffn_dim": 2048,
        "decoder_ffn_dim": 2048,
        "encoder_attention_heads": 12,
        "decoder_attention_heads": 12,
        "d_model": 768,
        "activation_function": "gelu",
        "scale_embedding": True,
        "add_final_layer_norm": True,
        "share_encoder_decoder_embeddings": False,
        "tokenizer_max_length": 64,

        "learning_rate": 7e-5,
        "batch_size": 128,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine_with_restarts",
        "warmup_steps": 500,
        "optim": "adamw_torch",
        "max_grad_norm": 0.8,
        "pad_to_multiple_of": 8,

        "eval_steps": 2000,
        "save_steps": 2000,
        "save_total_limit": 1,
        "early_stopping_patience": 25,

        "test_samples": [
            ">>abk<< Ацгәы аԥенџьыр",
            ">>rus<< Она приехала вместе с ним на этой машине",
            ">>rus<< И кошки и собаки были довольны",
            ">>abk<< Ари ашәҟәы сара изгар истахуп",
            ">>rus<< Ну зачем, Фрэнк, ты так сделал?",
        ]
    }

    logger.info("Starting training process...")
    try:
        train_model(first_run_config, resume_from_checkpoint=True)
    except Exception as e:
        logger.exception(f"Training failed: {e}")

if __name__ == "__main__":
    main()