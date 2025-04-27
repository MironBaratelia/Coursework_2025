import os
import random
import logging
import string
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

punct_chars = string.punctuation + '«»—''""‹›„‚'
translator = str.maketrans('', '', punct_chars)

def remove_punctuation(text):
    return text.translate(translator).strip()

def load_tsv_pairs(data_dir):
    pairs = []
    tsv_files = [
        "translations_1.tsv",
        "translations_2.tsv",
        "translations_3.tsv",
        "words.tsv",
        "phrases.tsv"
    ]
    
    for file_name in tsv_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    src = parts[0].strip()
                    tgt = parts[1].strip()
                    pairs.append((src, tgt))
    
    return pairs

def generate_combined_sample(pairs):
    if not pairs:
        return ("", "")
    
    n = random.randint(1, 15)
    selected_pairs = random.choices(pairs, k=n)
    
    separators = []
    for _ in range(n-1):
        if random.random() < 0.4:
            sep = ' '
        else:
            sep = random.choice([', ', '; ', '\n'])
        separators.append(sep)
    
    src_parts = []
    tgt_parts = []
    
    for i, (src, tgt) in enumerate(selected_pairs):
        src_parts.append(src)
        tgt_parts.append(tgt)
        
        if i < len(separators):
            src_parts.append(separators[i])
            tgt_parts.append(separators[i])
    
    return (
        ''.join(src_parts).strip(),
        ''.join(tgt_parts).strip()
    )

def augment_line(src, tgt):
    tgt = tgt[0].upper() + tgt[1:] if len(tgt) > 0 else tgt
    versions = []
    
    is_question = src.endswith('?') or tgt.endswith('?')
    last_char = src[-1] if len(src) > 0 else ''
    needs_punct = last_char in {'.', '!'} or last_char.isalpha()
    
    if is_question:
        base = src.rstrip('?')
        versions.append((base, tgt.rstrip('?') + '?'))
        versions.append((base + '?', tgt.rstrip('?') + '?'))
    elif needs_punct:
        base = src.rstrip('.!')
        versions.extend([
            (base, tgt.rstrip('.!')),
            (base + '.', tgt.rstrip('.!') + '.'),
            (base + '!', tgt.rstrip('.!') + '!')
        ])
    else:
        versions.append((src, tgt))

    transforms = [
        (str.lower, str.lower),
        (str.capitalize, str.capitalize),
        (lambda x: x[0].upper() + x[1:] if len(x) > 0 else x, lambda x: x[0].upper() + x[1:] if len(x) > 0 else x),
        (str.title, str.title),
        (str.title, str),
        (str.capitalize, str),
        (lambda x: x.capitalize().lower(), str.lower),
    ]
    
    for src_tr, tgt_tr in transforms:
        try:
            src1 = src_tr(src)
            tgt1 = tgt_tr(tgt)
            versions.append((src1, tgt1))
        except:
            continue

    return versions

class BilingualDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=64, is_val=False, add_synthetic=False):
        self.lang_to_id = {'abk': 0, 'rus': 1}
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_val = is_val
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.tsv') and ('val' in file) == is_val:
                    file_path = os.path.join(root, file)
                    if file in ['corpus_abkhaz.tsv', 'corpus_russian.tsv']:
                        self.process_file(file_path)
                    else:
                        self.process_file(file_path)
                        if any(keyword in file for keyword in {"100text", "phrases", "words", "main", "ru_worlds", "corrections", "number"}):
                            for i in range(3):
                                self.process_file(file_path)
        
        if add_synthetic and not is_val:
            logger.info("Generating synthetic examples from word combinations...")
            pairs = load_tsv_pairs(data_dir)
            if pairs:
                for _ in tqdm(range(300000)):
                    abk, rus = generate_combined_sample(pairs)
                    if abk and rus:
                        self.add_pair(abk, rus, src_lang='abk', tgt_lang='rus')
                        self.add_pair(rus, abk, src_lang='rus', tgt_lang='abk')
        
        random.shuffle(self.samples)
        logger.info(f"Loaded {len(self.samples)} examples")

    def process_file(self, file_path):
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                abk, rus = parts[0].strip(), parts[1].strip()
                
                if file_name == 'corpus_abkhaz.tsv':
                    self.add_pair(rus, abk, src_lang='rus', tgt_lang='abk')
                        
                elif file_name == 'corpus_russian.tsv':
                    self.add_pair(abk, rus, src_lang='abk', tgt_lang='rus')

                elif file_name == 'low_quality_russian.tsv':
                    rn = random.random()
                    if rn < 0.1 and len(rus) > 20:
                        if rn < 0.04:
                            self.add_pair(rus.capitalize().lower(), rus.lower(), src_lang='abk', tgt_lang='rus')
                        elif rn < 0.08:
                            self.add_pair(rus.capitalize(), rus, src_lang='abk', tgt_lang='rus')
                        else:
                            self.add_pair(rus, rus, src_lang='abk', tgt_lang='rus')
                            
                elif file_name == 'low_quality_abkhaz.tsv':
                    rn = random.random()
                    if rn < 0.15 and len(abk) > 20:
                        if rn < 0.06:
                            self.add_pair(abk.capitalize().lower(), abk.lower(), src_lang='rus', tgt_lang='abk')
                        elif rn < 0.12:
                            self.add_pair(abk.capitalize(), abk, src_lang='rus', tgt_lang='abk')
                        else:
                            self.add_pair(abk, abk, src_lang='rus', tgt_lang='abk')
                else:
                    if self.is_val:
                        self.add_pair(abk, rus, src_lang='abk', tgt_lang='rus')
                        self.add_pair(rus, abk, src_lang='rus', tgt_lang='abk')
                    else:
                        for augmented_src, original_tgt in augment_line(abk, rus):
                            self.add_pair(augmented_src, original_tgt, src_lang='abk', tgt_lang='rus')
                        for augmented_src, original_tgt in augment_line(rus, abk):
                            self.add_pair(augmented_src, original_tgt, src_lang='rus', tgt_lang='abk')

    def add_pair(self, src_text, tgt_text, src_lang, tgt_lang):
        self.samples.append({
            "src": f">>{src_lang}<< {src_text}",
            "tgt": f">>{tgt_lang}<< {tgt_text}",
            "target_lang": tgt_lang
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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

        if len(src_enc['input_ids'][0]) > self.max_length or len(tgt_enc['input_ids'][0]) > self.max_length:
            return self[(idx + 1) % len(self)]

        return {
            "input_ids": src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels": tgt_enc["input_ids"].squeeze(),
            "target_lang": self.lang_to_id[sample["target_lang"]]
        } 