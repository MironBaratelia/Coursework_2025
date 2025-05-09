import torch
import wandb
from transformers.trainer_callback import TrainerCallback
from config import CONFIG

def tokenize_dataset(dataset, tokenizer, max_length, verbose=False):
    def tokenize_function(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        result = tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=None, 
        remove_columns=dataset.column_names,
        desc="токенизация",
        disable=not verbose
    )
            
    return tokenized_dataset

class ModelEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_steps=2000, test_phrase="Бзиара умаз"):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.test_phrase = test_phrase
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            model = kwargs.get("model")

            print(f"\n--- Тест на шаге {state.global_step} ---")
            print(f"Запрос: '{self.test_phrase}'")
            
            messages = [{"role": "user", "content": self.test_phrase}]
            try:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text, return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                
                original_training_state = model.training
                model.eval()
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9,
                    )
                    output_ids = generated_ids[0][len(input_ids[0]):].tolist()
                    response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                    print(f"Ответ: '{response}'")
                    
                    if CONFIG["use_wandb"] and wandb.run is not None:
                        wandb.log({
                            "eval/step": state.global_step,
                            "eval/test_phrase": self.test_phrase,
                            "eval/response": response,
                        })
            except Exception as e:
                print(f"Error: {e}")
            finally:
                if original_training_state:
                    model.train()