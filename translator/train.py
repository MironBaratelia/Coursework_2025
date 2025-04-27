import torch
import os
import numpy as np
import logging
import wandb
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast
)
import sacrebleu

from model import initialize_model
from data_utils import BilingualDataset
from callbacks import TranslationCallback, DatasetRebuildCallback

logger = logging.getLogger(__name__)

def train_model(config):
    wandb.init(config=config, project=config.get("project_name", "apsua-translator"))
    wandb.config.update(config)

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
    tokenizer.model_max_length = 64

    model = initialize_model(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class CustomDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features):
            for f in features:
                f.pop("target_lang", None)
            return super().__call__(features)

    data_collator = CustomDataCollator(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )
    
    train_dataset = BilingualDataset(data_dir, tokenizer, add_synthetic=True)
    val_dataset = BilingualDataset(data_dir, tokenizer, is_val=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        cleaned_preds = []
        cleaned_refs = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            clean_pred = pred.strip()
            clean_label = label.strip()
            
            cleaned_preds.append(clean_pred)
            cleaned_refs.append([clean_label])
        
        metrics = {}
        if cleaned_preds:
            bleu = sacrebleu.corpus_bleu(
                cleaned_preds,
                cleaned_refs,
                smooth_method='add-k',
                smooth_value=1
            )
            metrics['bleu'] = bleu.score
        else:
            metrics['bleu'] = 0.0
        
        return metrics

    test_samples = config.get("test_samples", [
        ">>abk<< Ацгәы аԥенџьыр",
        ">>rus<< Она приехала вместе с ним на этой машине"
    ])

    training_args = Seq2SeqTrainingArguments(
        max_grad_norm=wandb.config.max_grad_norm,
        output_dir=output_dir,
        learning_rate=wandb.config.learning_rate,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=config.get("warmup_steps", 1000),
        fp16_full_eval=True,
        per_device_train_batch_size=wandb.config.batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        num_train_epochs=config.get("num_train_epochs", 3),
        logging_dir=config.get("logging_dir", "./logs"),
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=wandb.config.eval_steps,
        save_strategy="steps",
        save_steps=wandb.config.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        weight_decay=wandb.config.weight_decay,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            TranslationCallback(tokenizer, test_samples, device)
        ]
    )
    
    dataset_rebuild_callback = DatasetRebuildCallback(data_dir, tokenizer, trainer)
    trainer.add_callback(dataset_rebuild_callback)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    wandb.finish() 