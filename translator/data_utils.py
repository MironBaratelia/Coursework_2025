import os
import random
import logging
from typing import List, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

def augment_line(line: str) -> List[Tuple[str, str]]:
    parts = line.strip().split('\\t')
    if len(parts) != 2:
        return []

    abk, rus = parts[0].strip(), parts[1].strip()
    versions = []

    transforms = [
        (str.lower, str.lower),
        (str.lower, str.lower),
        (lambda x: x.capitalize(), lambda x: x.capitalize()),
        (lambda x: x[0].upper() + x[1:] if len(x) > 0 else x, lambda x: x[0].upper() + x[1:] if len(x) > 0 else x),
        (lambda x: x.title(), lambda x: x.title()),
    ]

    for i, (abk_tr, rus_tr) in enumerate(transforms):
        try:
            src = abk_tr(abk)
            tgt = rus_tr(rus)

            if i == 2:
                base_src = src.rstrip('.!')
                base_tgt = tgt.rstrip('.!')
                endings = ['', '.', '!']

                for ending in endings:
                    variant_src = base_src + ending
                    variant_tgt = base_tgt + ending
                    if len(variant_src) > 1 and len(variant_tgt) > 1:
                        versions.append((variant_src, variant_tgt))
            else:
                if len(src) > 1 and len(tgt) > 1:
                    versions.append((src, tgt))
        except IndexError:
            logger.warning(f"Skipping transformation due to short string: abk='{abk}', rus='{rus}'")
            continue

    return versions

class BilingualDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 64, is_val: bool = False):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_val = is_val
        self.processed_files_count = 0
        self.augmented_files_multiplier = 3

        important_files_keywords = {"dialog_my", "dialog_2", "phrases", "words", "main", "simple", "ru_worlds", "corrections", "number"}

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.tsv') and (('val' in file) == is_val):
                    file_path = os.path.join(root, file)
                    self.process_file(file_path)
                    if not is_val and any(keyword in file for keyword in important_files_keywords):
                        for _ in range(self.augmented_files_multiplier):
                             self.process_file(file_path, use_augmentation=True)

        random.shuffle(self.samples)
        logger.info(f"Loaded {len(self.samples)} examples from {self.processed_files_count} files ({'validation' if is_val else 'training'}).")

    def process_file(self, file_path: str, use_augmentation: bool = True):
        self.processed_files_count +=1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if self.is_val:
                        parts = line.strip().split('\\t')
                        if len(parts) == 2:
                            src, tgt = parts[0].strip(), parts[1].strip()
                            self.add_pair(src, tgt)
                    elif use_augmentation:
                        for src, tgt in augment_line(line):
                             self.add_pair(src, tgt)
                    else:
                         parts = line.strip().split('\\t')
                         if len(parts) == 2:
                             src, tgt = parts[0].strip(), parts[1].strip()
                             self.add_pair(src, tgt)

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")


    def add_pair(self, src: str, tgt: str):
        if len(src) > 0 and len(tgt) > 0:
            self.samples.append({
                "src": f">>abk<< {src}",
                "tgt": tgt,
                "target_lang": "rus"
            })
            self.samples.append({
                "src": f">>rus<< {tgt}",
                "tgt": src,
                "target_lang": "abk"
            })
        else:
            logger.warning(f"Skipping empty pair: src='{src}', tgt='{tgt}'")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        src_enc = self.tokenizer(
            sample["src"],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                sample["tgt"],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )

        if src_enc["input_ids"].shape[-1] > self.max_length or \
           tgt_enc["input_ids"].shape[-1] > self.max_length:
             logger.warning(f"Tokenized length exceeds max_length for sample index {idx}. Skipping.")
             return self[(idx + 1) % len(self)]

        return {
            "input_ids": src_enc["input_ids"].squeeze(0),
            "attention_mask": src_enc["attention_mask"].squeeze(0),
            "labels": tgt_enc["input_ids"].squeeze(0)
        } 