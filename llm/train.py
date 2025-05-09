import torch
import os
import time
import wandb
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import set_seed, default_data_collator

from config import CONFIG
from data_utils import load_and_process_files
from model import tokenize_dataset, ModelEvaluationCallback

set_seed(CONFIG["seed"])

if os.name == 'nt':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

torch.backends.cudnn.benchmark = True

def train():
    run_name = f"qwen-abkhaz-{int(time.time())}"
    effective_use_wandb = CONFIG["use_wandb"]

    if effective_use_wandb:
        try:
            wandb.init(project="qwen-abkhaz-finetune", name=run_name, config=CONFIG)
        except Exception as e:
            print(f"Error initializing W&B: {e}. Disabling W&B.")
            effective_use_wandb = False
    
    try:
        dataset = load_and_process_files(verbose=False)
        
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["custom_tokenizer_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenized_dataset = tokenize_dataset(
            dataset, 
            tokenizer, 
            max_length=CONFIG["max_length"], 
            verbose=False
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.bfloat16, 
            device_map="auto",
        )
        
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            embedding_layer = model.get_input_embeddings()
            with torch.no_grad():
                for token_id in range(model.config.vocab_size, len(tokenizer)):
                    embedding_layer.weight[token_id].normal_(mean=0.0, std=0.02)
        
        training_args = TrainingArguments(
            output_dir=CONFIG["output_dir"],
            run_name=run_name,
            per_device_train_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            num_train_epochs=CONFIG["epochs"],
            learning_rate=CONFIG["learning_rate"],
            logging_steps=CONFIG["logging_steps"],
            logging_strategy="steps",
            logging_dir=f"{CONFIG['output_dir']}/logs",
            save_steps=CONFIG["save_steps"],
            save_strategy="steps",
            warmup_ratio=CONFIG["warmup_ratio"],
            weight_decay=CONFIG["weight_decay"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            bf16=True, 
            fp16=False,
            dataloader_num_workers=0, 
            dataloader_pin_memory=True, 
            group_by_length=False, 
            save_total_limit=2,
            report_to="wandb" if effective_use_wandb else "none",
            optim="adamw_torch", 
            gradient_checkpointing=True, 
            seed=CONFIG["seed"],
            ddp_find_unused_parameters=False,
        )
        
        if effective_use_wandb and wandb.run is not None:
            wandb.watch(model, log="gradients", log_freq=CONFIG["logging_steps"])
        
        evaluation_callback = ModelEvaluationCallback(
            tokenizer=tokenizer,
            eval_steps=CONFIG["eval_steps"]
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=default_data_collator,
            callbacks=[evaluation_callback],
        )
        
        resume_checkpoint = None
        if CONFIG["resume_from_checkpoint"]:
            output_dir = CONFIG["output_dir"]
            checkpoints = [
                folder for folder in os.listdir(output_dir)
                if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, folder))
            ]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint_name = checkpoints[-1]
                latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint_name)
                resume_checkpoint = latest_checkpoint_path
                print(f"Чекпоинт найден: {latest_checkpoint_path}")
            else:
                print("Чекпоинт не найден")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        final_model_path = os.path.join(CONFIG["output_dir"], "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model {final_model_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
    
    finally:
        wandb.finish()