import torch
import logging
from typing import List
from transformers import TrainerCallback, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class TranslationCallback(TrainerCallback):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, examples: List[str], device: str = "cuda"):
        self.tokenizer = tokenizer
        self.examples = examples
        self.device = device
        self.log_frequency_steps = 20000

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_frequency_steps == 0 and model is not None:
            logger.info(f"\n--- Translation Examples at Step {state.global_step} ---")
            model.eval()
            with torch.no_grad():
                for example in self.examples:
                    try:
                        inputs = self.tokenizer(example, return_tensors="pt").to(self.device)
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=64,
                            num_beams=5,
                            early_stopping=True,
                            length_penalty=1.05,
                            repetition_penalty=1.05,
                        )

                        input_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                        logger.info(f"Input:  {input_text}")
                        logger.info(f"Output: {output_text}")

                    except Exception as e:
                        logger.error(f"Error during translation callback for example '{example}': {e}")
            logger.info(f"--- End Translation Examples ---\n")
            model.train() 