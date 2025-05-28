import os
import json
from tqdm import tqdm


def load_jsonl_file(filepath: str):
    """
    加载 JSONL 文件并返回条目列表。
    
    :param filepath: JSONL 文件路径。
    :return: 每行 JSON 对象组成的列表。
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_fields(entries: list[dict], fields: list[str]):
    """
    提取 JSON 条目中的指定字段。
    
    :param entries: JSON 条目列表。
    :param fields: 要提取的字段列表。
    :return: 包含指定字段的新条目列表。
    """
    extracted_data = []
    for entry in tqdm(entries, desc="loading dataset"):
        if all(field in entry for field in fields):  # 确保所有字段都存在
            extracted_data.append({field: entry[field] for field in fields})
        else:
            missing_fields = [field for field in fields if field not in entry]
            print(f"Skipping entry due to missing fields: {missing_fields}")
    return extracted_data


def read_dataset(data_type: str, input_path):
    # 定义不同数据类型的字段规则
    data_type_to_fields = {
        "MATH": ["problem", "solution", "answer", "subject"],
        "GSM8K": ["question", "solution", "answer"]
    }

    if data_type not in data_type_to_fields:
        print(f"Unsupported data type: {data_type}")
        return []
    if data_type == "MATH":
        print(f"You are loading {data_type} dataset")
        input_path = input_path
    elif data_type == "GSM8K":
        print(f"You are loading {data_type} dataset")
        input_path = input_path
    # 加载 JSONL 数据
    entries = load_jsonl_file(input_path)

    # 提取指定字段
    fields = data_type_to_fields[data_type]
    extracted_data = extract_fields(entries, fields)

    return extracted_data