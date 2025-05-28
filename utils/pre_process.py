import re
import itertools
from tqdm import tqdm
import copy
from utils.constants import *
import nltk
import os
import json
from nltk.tokenize import sent_tokenize
from utils.read_prompt import *
nltk.data.path.append('add your nltk/tokenizer/punkt data path if needed')

def process_and_save_data(prompt_path, dataset_name, input_path):
    print("Loading prompt")
    yaml_data = read_yaml(prompt_path)
    print("Loading dataset")
    train_data = read_dataset(dataset_name, input_path=input_path)
    print("Merging data and prompt")
    train_data = merge_data_and_prompt(data=train_data, prompt=yaml_data)
    print("Saving processed data")
    processed_path = save_to_jsonl(train_data, input_path)
    return processed_path
    
    
def load_jsonl_file(filepath: str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_fields(entries: list[dict], fields: list[str]):
    extracted_data = []
    for entry in tqdm(entries, desc="loading dataset"):
        if all(field in entry for field in fields):  # 确保所有字段都存在
            extracted_data.append({field: entry[field] for field in fields})
        else:
            missing_fields = [field for field in fields if field not in entry]
            print(f"Skipping entry due to missing fields: {missing_fields}")
    return extracted_data


def read_dataset(data_type: str, input_path):
    data_type_to_fields = {
        "MATH": ["idx", "MATH_PROBLEM", "REASONING_STEP", "ANSWER", "DEBATE_STEP", "speech_history", "judge_decision"],
    }

    if data_type not in data_type_to_fields:
        print(f"Unsupported data type: {data_type}")
        return []
    if data_type == "MATH":
        print(f"You are loading {data_type} dataset")
    entries = load_jsonl_file(input_path)

    fields = data_type_to_fields[data_type]
    extracted_data = extract_fields(entries, fields)

    return extracted_data
    
def extract_sentence(text):

    sentences = sent_tokenize(text)
    
    # 返回拆分后的句子
    return sentences


def replace_placeholders(data, replacements):
    """
    递归替换字典或列表中的占位符。
    """
    if isinstance(data, dict):
        return {key: replace_placeholders(value, replacements) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(item, replacements) for item in data]
    elif isinstance(data, str):
        pattern = re.compile('|'.join(map(re.escape, replacements.keys())))
        return pattern.sub(lambda match: replacements[match.group(0)], data)
    else:
        return data


def merge_data_and_prompt(data, prompt):
    
    prompts_ = []
    for item in tqdm(data, desc="merging data and prompt"):
        # ["idx", "MATH_PROBLEM", "REASONING_STEP", "ANSWER", "DEBATE_STEP", "speech_history", "judge_decision"]
        idx = item.get("idx", "No idx provided")
        math_problem = item.get("MATH_PROBLEM", "No MATH_PROBLEM provided")
        reasoning_step = item.get("REASONING_STEP", "No REASONING_STEP provided")
        answer = item.get("ANSWER", "No ANSWER provided")
        debate_step = item.get("DEBATE_STEP", "No DEBATE_STEP provided")
        speech_history = item.get("speech_history", "No speech_history provided")
        judge_decision = item.get("judge_decision", "No judge_decision provided")
        # solution_sentence = extract_sentence(text=solution)
        # for sentence in solution_sentence:
        replacements = {
            "<PROBLEM>": math_problem,
            "<RS>": reasoning_step,
            "<Single Step>": debate_step,
            "<ANSWER>": answer
        }
        updated_data = replace_placeholders(prompt, replacements)
            
        rules = "\n".join(updated_data["Debate_Prompt"]["Rules"])
            
            
        debater_a = updated_data["Debate_Prompt"]["Debater_A"]
        debater_b = updated_data["Debate_Prompt"]["Debater_B"]
        judge = updated_data["Debate_Prompt"]["Judge"]["description"]
        
        A_ = rules + debater_a['description']
        B_ = rules + debater_b['description']
        judge_ = rules + judge
        
        Response_A = speech_history['Debater_A']
        Response_B = speech_history['Debater_B']
        Response_judge = judge_decision
        
        prompt_ = {
            "A_prompt": A_,
            "B_prompt": B_,
            "judge_prompt": judge_,
            "Response_A": Response_A,
            "Response_B": Response_B,
            "Response_judge": Response_judge
        }
        
        prompts_.append(prompt_)
    
    return prompts_
    
    
def save_to_jsonl(data, file_path):
    directory, filename = os.path.split(file_path)
    
    name, ext = os.path.splitext(filename)
    processed_filename = f"{name}_processed{ext}"
    
    processed_file_path = os.path.join(directory, processed_filename)
    
    if isinstance(data, dict):
        data = [data]
    with open(processed_file_path, "w", encoding="utf-8") as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")
    
    print(f"Data saved to {processed_file_path}")
    return processed_file_path