import torch
import os
import logging
import wandb
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    EarlyStoppingCallback
)

from data_utils import BilingualDataset
from model import initialize_model
from callbacks import TranslationCallback

logger = logging.getLogger(__name__)

def train_model(config, project_name="apsua-translator", resume_from_checkpoint=True):
    wandb.init(config=config, project=project_name, resume="allow")

    data_dir = config.get("data_dir", "1data/translate_t")
    tokenizer_path = config.get("tokenizer_path", "tokenizer")
    output_dir = config.get("output_dir", "./apsua_translator")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        bos_token=">>lang<<",
        additional_special_tokens=[">>abk<<", ">>rus<<"]
    )
    tokenizer.model_max_length = config.get("tokenizer_max_length", 64)

    model = initialize_model(
        tokenizer,
        dropout=wandb.config.dropout,
        max_position_embeddings=config.get("max_position_embeddings", 64),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=config.get("pad_to_multiple_of", 8)
    )

    logger.info("Loading training dataset...")
    train_dataset = BilingualDataset(data_dir, tokenizer, max_length=tokenizer.model_max_length)
    logger.info("Loading validation dataset...")
    val_dataset = BilingualDataset(data_dir, tokenizer, max_length=tokenizer.model_max_length, is_val=True)

    test_samples = config.get("test_samples", [
        ">>abk<< Ацгәы аԥенџьыр",
        ">>rus<< Она приехала вместе с ним на этой машине",
        ">>rus<< И кошки и собаки были довольны",
        ">>abk<< Ари ашәҟәы сара изгар истахуп",
        ">>rus<< Ну зачем, Фрэнк, ты так сделал?",
    ])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=wandb.config.learning_rate,
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine_with_restarts"),
        warmup_steps=config.get("warmup_steps", 500),
        per_device_train_batch_size=wandb.config.batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        num_train_epochs=config.get("num_train_epochs", 5),
        logging_dir=config.get("logging_dir", "./logs"),
        logging_steps=config.get("logging_steps", 100),
        evaluation_strategy="steps",
        eval_steps=wandb.config.eval_steps,
        save_strategy="steps",
        save_steps=wandb.config.eval_steps,
        save_total_limit=config.get("save_total_limit", 1),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=config.get("optim", "adamw_torch"),
        weight_decay=wandb.config.weight_decay,
        fp16=torch.cuda.is_available(),
        max_grad_norm=wandb.config.max_grad_norm,
        report_to="wandb",
        remove_unused_columns=False,
    )

    translation_callback = TranslationCallback(tokenizer, test_samples, device)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.get("early_stopping_patience", 25)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[translation_callback, early_stopping_callback]
    )

    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except FileNotFoundError:
         logger.warning("No checkpoint found, starting training from scratch.")
         trainer.train(resume_from_checkpoint=False)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    wandb.finish()

    logger.info("Training finished.") 