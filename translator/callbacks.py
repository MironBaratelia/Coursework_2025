import torch
import logging
from transformers import TrainerCallback
from data_utils import BilingualDataset

logger = logging.getLogger(__name__)

class TranslationCallback(TrainerCallback):
    def __init__(self, tokenizer, examples, device="cuda"):
        self.tokenizer = tokenizer
        self.examples = examples
        self.device = device

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 10000 == 0 and model is not None:
            model.eval()
            with torch.no_grad():
                for example in self.examples:
                    inputs = self.tokenizer(example, return_tensors="pt").to(self.device)
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=64,
                        num_beams=5,
                        early_stopping=False,
                        length_penalty=1.15,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=2,
                        encoder_repetition_penalty=1.3
                    )
                    input_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0], skip_special_tokens=False)
                    output_tokens = self.tokenizer.convert_ids_to_tokens(outputs[0], skip_special_tokens=False)
                    
                    logger.info(f"\nStep {state.global_step}")
                    logger.info(f"Input tokens: {input_tokens}")
                    logger.info(f"Output tokens: {output_tokens}\n")
            model.train()
        
class DatasetRebuildCallback(TrainerCallback):
    def __init__(self, data_dir, tokenizer, trainer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.epoch = 0
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch += 1
        logger.info(f"Epoch {self.epoch} completed. Rebuilding dataset...")
        
        new_train_dataset = BilingualDataset(self.data_dir, self.tokenizer, add_synthetic=True)
        
        self.trainer.train_dataset = new_train_dataset
        
        logger.info(f"Dataset rebuilt with {len(new_train_dataset)} examples for epoch {self.epoch+1}") 