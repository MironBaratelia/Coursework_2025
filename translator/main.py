import logging
import os
from train import train_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    first_run_config = {
        "learning_rate": 7e-5,
        "batch_size": 128,
        "weight_decay": 0.01,
        "dropout": 0.2,
        "gradient_accumulation_steps": 1,
        "eval_steps": 2500,
        "max_grad_norm": 0.8,
        "warmup_steps": 1000,
        "num_train_epochs": 3,
        "data_dir": "1data/translate_t",
        "tokenizer_path": "tokenizer",
        "output_dir": "./apsua_translator",
        "logging_dir": "./logs",
        "project_name": "apsua-translator",
        "test_samples": [
            ">>abk<< Ацгәы аԥенџьыр",
            ">>rus<< Она приехала вместе с ним на этой машине",
            ">>rus<< И кошки и собаки были довольны",
            ">>abk<< Ари ашәҟәы сара изгар исҭахуп",
            ">>rus<< Ну зачем, Фрэнк, ты так сделал?",
        ]
    }
    
    logger.info("Starting training process...")
    train_model(first_run_config)