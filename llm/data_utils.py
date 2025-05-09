import os
import glob
import json
import tqdm
from datasets import Dataset, concatenate_datasets
from config import CONFIG

def extract_text_from_data(data):
    messages = []
    if isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            valid_roles = ["system", "user", "assistant"]
            for msg in data["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg and msg["role"] in valid_roles:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            return messages if messages else None
        
        elif "conversation" in data and isinstance(data["conversation"], list):
            valid_roles = ["system", "user", "assistant"]
            for msg in data["conversation"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg and msg["role"] in valid_roles:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            return messages if messages else None
        
        elif "text" in data and isinstance(data["text"], str) and data["text"].strip():
            messages.append({"role": "assistant", "content": data["text"].strip()})
            return messages
        
        elif "role" in data and "content" in data:
            valid_roles = ["system", "user", "assistant"]
            if data["role"] in valid_roles:
                messages.append({"role": data["role"], "content": data["content"]})
            return messages if messages else None
        
    elif isinstance(data, list):
        valid_roles = ["system", "user", "assistant"]
        for item in data:
            if isinstance(item, dict) and "role" in item and "content" in item and item["role"] in valid_roles:
                messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, str) and item.strip():
                messages.append({"role": "assistant", "content": item.strip()})
        return messages if messages else None
    
    elif isinstance(data, str) and data.strip():
        messages.append({"role": "assistant", "content": data.strip()})
        return messages
    return None

def load_and_process_files(verbose=False):
    data_files = []
    for dir_path in CONFIG["data_dirs"]:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files = glob.glob(f"{dir_path}/*.jsonl")
            if files:
                data_files.extend(files)

    if not data_files:
        raise ValueError(f"No .jsonl files found in {CONFIG['data_dirs']}")
    
    datasets_list = []
    total_examples_from_files = 0
    
    for file_path in tqdm.tqdm(data_files, desc="Loading & Processing Files", disable=not verbose):
        valid_examples = []
        max_lines = CONFIG["max_lines_per_file"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_lines is not None and i >= max_lines:
                        break
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            messages = extract_text_from_data(data)
                            if messages:
                                valid_examples.append({"messages": messages})
                        except Exception:
                            pass 
            
            if valid_examples:
                dataset_item = Dataset.from_list(valid_examples)
                basename = os.path.basename(file_path)
                
                if basename.startswith("formatted"):
                    multiplier = CONFIG["add_formatter_files_times"]
                    datasets_list.extend([dataset_item] * multiplier)
                
                datasets_list.append(dataset_item)
                total_examples_from_files += len(dataset_item)
        except Exception as e:
            if verbose:
                print(f"Error {file_path}: {e}")
    
    combined_dataset = concatenate_datasets(datasets_list)
    combined_dataset = combined_dataset.shuffle(seed=CONFIG["seed"])
    
    if verbose:
        print(f"Всего примеров: {len(combined_dataset)}")
            
    return combined_dataset